"""Incremental observed-token HYB-B candidate discovery for the locked Qwen BPE.

Only the first 4096 generated thinking tokens can produce opportunities. Word
boundaries and spacing match the offline annotator; k=4 counts the merged,
ordered candidate stream. The 4096 point is always the PUB-gated BF endpoint.
"""
from bisect import bisect_left
import codecs
import time
from exec_exit.analysis.trajectory_annotations import WORD_PATTERN


class HybridPositions:
    def __init__(self,codec,*,close_id=151668,eos_ids=(151643,151645)):
        self.codec=codec;self.stop_ids={close_id,*eos_ids};self.processed=0;self.thinking=0
        self.text="";self.byte_ends=[];self.char_byte_ends=[];self.bytes=0;self.chars_bytes=0
        self.decoder=codecs.getincrementaldecoder("utf-8")("strict");self.last_word=None
        self.seen_matches=set();self.candidates=set();self.emitted=set();self.stopped=False;self.invalid=False
        self.calls=0;self.cpu_ns=0;self.max_prefix_tokens=0

    def __call__(self,ids,finished=False):
        began=time.thread_time_ns();self.calls+=1
        try:
            if len(ids)<self.processed:raise ValueError("Main token stream cannot shrink")
            if self.stopped:self.processed=len(ids);return []
            old_chars=len(self.text)
            for token in ids[self.processed:]:
                self.processed+=1
                if token in self.stop_ids:self.stopped=True;break
                self.thinking+=1
                if self.thinking>4096:self.stopped=True;break
                if self.thinking in (2048,4096):self.candidates.add(self.thinking)
                try:
                    part=self.codec.vocab[token];self.bytes+=len(part);self.byte_ends.append(self.bytes)
                    if not self.invalid:
                        decoded=self.decoder.decode(part,final=False);self.text+=decoded
                        for ch in decoded:self.chars_bytes+=len(ch.encode("utf-8"));self.char_byte_ends.append(self.chars_bytes)
                except (KeyError,UnicodeDecodeError):self.invalid=True
            self.max_prefix_tokens=max(self.max_prefix_tokens,min(self.thinking,4096))
            if not self.invalid:
                # Largest match is Alternatively (13 chars); retain its left
                # boundary as well. Match coordinates remain global.
                for match in WORD_PATTERN.finditer(self.text,max(0,old_chars-15)):
                    key=(match.start(),match.end())
                    if key in self.seen_matches or match.end()==len(self.text):continue
                    self.seen_matches.add(key)
                    p=bisect_left(self.byte_ends,self.char_byte_ends[match.end()])+1
                    if p>4096:continue
                    if self.last_word is not None and p-self.last_word<1024:continue
                    self.last_word=p;self.candidates.add(p)
            selected=set(sorted(self.candidates)[::4])
            if 4096 in self.candidates:selected.add(4096)
            result=sorted(selected-self.emitted)
            if self.emitted and result and min(result)<max(self.emitted):raise ValueError("New opportunity would revise already-visible order")
            self.emitted.update(result)
            if len(self.candidates)>6 or len(self.emitted)>3:raise AssertionError("HYB-B bounded candidate proof violated")
            return result
        finally:self.cpu_ns+=time.thread_time_ns()-began

    def metadata(self):
        return {"calls":self.calls,"host_thread_cpu_ns":self.cpu_ns,"max_inspected_thinking_tokens":self.max_prefix_tokens,
            "merged_candidates":sorted(self.candidates),"selected_hyb_b_positions":sorted(self.emitted),
            "text_status":"UNKNOWN_INVALID_UTF8_OR_TOKEN" if self.invalid else "KNOWN",
            "future_tokens_read":False,"retokenized":False}
