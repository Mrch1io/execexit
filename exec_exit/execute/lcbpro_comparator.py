"""Statically reviewed LCB Pro ordinary checkers; no checker is executed.

PUB and post-freeze scoring call this same function with the same per-task spec.
Semantics are derived from the pinned checker sources and LightCPVerifier's
pinned testlib header, then checked with public-answer variants. This is not a
dynamic equivalence claim. No package/test loader or model code exists here.
"""
from __future__ import annotations
import fcntl,hashlib,json,re,time
from pathlib import Path
from exec_exit.execute import sandbox as original

TESTLIB_SHA256='70f74c570f2b45d63086ae6e4c41bbb5c5ffd4428cba9914bbc0396d29be10d8'
TESTLIB_REVISION='f6637f470f7c3d7d879986dc6ee057cdc5e7b0ac'
KINDS={'integer_tokens','literal_tokens','line_tokens','yesno_tokens'}
TOKEN_BLANKS=b' \t\r\n'  # testlib isBlanks(), not Python str.split().
CLASSIC_SPACE=b' \t\r\n\v\f'  # C-locale stringstream within an lcmp line.
MAX_TOKEN_BYTES=32*1024**2
_CANONICAL_INT=re.compile(rb'(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z')

class ComparatorSpecError(ValueError):pass
class ComparatorReferenceError(ValueError):pass
class ComparatorObservationError(ValueError):pass
class OutputSyntaxError(ValueError):pass


def validate_spec(spec):
    if not isinstance(spec,dict) or spec.get('kind') not in KINDS:raise ComparatorSpecError('Explicit statically supported comparator kind required')
    if not isinstance(spec.get('task_id'),str) or not spec['task_id']:raise ComparatorSpecError('Per-task comparator identity required')
    if not re.fullmatch('[0-9a-f]{64}',str(spec.get('checker_sha256',''))):raise ComparatorSpecError('Exact public checker SHA required')
    if spec.get('testlib_sha256')!=TESTLIB_SHA256:raise ComparatorSpecError('Different or unresolved testlib dependency')
    if spec['kind']=='yesno_tokens' and spec.get('count_rule') not in ('first_integer','sum_q_2118'):raise ComparatorSpecError('Explicit input-derived YESNO count required')
    if spec['kind']!='yesno_tokens' and spec.get('count_rule') not in (None,'none'):raise ComparatorSpecError('Count rule is only for YESNO family')
    return spec


def _bytes(value):
    if isinstance(value,str):value=value.encode('utf-8',errors='strict')
    if not isinstance(value,bytes):raise OutputSyntaxError('Output must be actual bytes or UTF8 text, never retokenized')
    # In this pinned testlib, EOFC is255 and buffered reader treats that byte
    # as EOF. Normal generated UTF8 cannot contain an isolated0xff byte.
    return value.split(b'\xff',1)[0]


def _token_iter(value):
    raw=_bytes(value);start=0
    for i,c in enumerate(raw):
        if c in TOKEN_BLANKS:
            if i>start:
                if i-start>MAX_TOKEN_BYTES:raise OutputSyntaxError('testlib token exceeds32MiB')
                yield raw[start:i]
            start=i+1
    if start<len(raw):
        if len(raw)-start>MAX_TOKEN_BYTES:raise OutputSyntaxError('testlib token exceeds32MiB')
        yield raw[start:]


def tokens(value):return list(_token_iter(value))


def parse_int64(token):
    # stringToLongLong rejects plus signs, leading zeroes, redundant minus,
    # NUL, decimals and exponents; std::stoll supplies signed64 range checking.
    if len(token)>20 or _CANONICAL_INT.fullmatch(token) is None:raise OutputSyntaxError('Noncanonical signed64 integer token')
    number=int(token)
    if not -(1<<63)<=number<(1<<63):raise OutputSyntaxError('Signed64 integer overflow')
    return number


def yesno_count(input_text,spec):
    if input_text is None:raise ComparatorReferenceError('YESNO requires the actual test input to determine answer count')
    try:
        values=iter(_token_iter(input_text))
        def take():
            try:return parse_int64(next(values))
            except StopIteration as exc:raise OutputSyntaxError('Truncated count-bearing input') from exc
        t=take()
        if t<0:raise OutputSyntaxError('Negative test-case count')
        if spec['count_rule']=='first_integer':return t
        total=0
        for _ in range(t):
            n=take();take() # k does not alter the output cardinality.
            if n<0:raise OutputSyntaxError('Invalid sum-q input structure')
            for _ in range(2*n):take()
            q=take()
            if q<0:raise OutputSyntaxError('Invalid query cardinality')
            for _ in range(q):take()
            total+=q
        return total
    except (OutputSyntaxError,UnicodeError) as exc:raise ComparatorReferenceError(str(exc)) from exc


def _read_string(raw,pos):
    """Pinned non-strict readStringTo/eoln, including its bare-CR behavior."""
    result=bytearray();end=len(raw)
    while pos<end:
        cur=raw[pos]
        if cur==10:break
        if cur==13:
            pos+=1
            if pos<end and raw[pos]==10:
                pos-=1;break
            # The original advances past a bare CR and appends the next char.
            result.append(raw[pos] if pos<end else 255)
            if pos<end:pos+=1
        else:result.append(cur);pos+=1
    if pos<end and raw[pos]==13 and pos+1<end and raw[pos+1]==10:pos+=2
    elif pos<end and raw[pos]==10:pos+=1
    return bytes(result),pos


def _line_words(line):
    # C/classic stringstream locale. Do not perform Unicode whitespace folding.
    return re.findall(rb'[^ \t\r\n\v\f]+',line)


def _compare_lines(actual,expected):
    ref=_bytes(expected);out=_bytes(actual);i=j=0
    while i<len(ref):
        line,i=_read_string(ref,i)
        if line==b'' and i==len(ref):break
        prediction,j=_read_string(out,j)
        if _line_words(line)!=_line_words(prediction):return False
    # quit(_ok) calls ouf.seekEof, whose whitespace set is narrower than
    # stringstream's. Extra nonblank tokens or VT/FF after consumed lines fail.
    return all(c in TOKEN_BLANKS for c in out[j:])


def reference_tokens(expected,spec,*,input_text=None):
    """Validate the standard answer; malformed references are not candidate FAIL."""
    validate_spec(spec)
    try:
        words=tokens(expected)
        if spec['kind']=='integer_tokens':return [parse_int64(w) for w in words]
        if spec['kind']=='yesno_tokens':
            lowered=[w.lower() for w in words]
            if any(w not in (b'yes',b'no') for w in lowered):raise OutputSyntaxError('Reference is not YES/NO')
            if len(words)!=yesno_count(input_text,spec):raise OutputSyntaxError('Reference YESNO count differs from public checker input-derived count')
            return lowered
        return words
    except (OutputSyntaxError,UnicodeError) as exc:raise ComparatorReferenceError(str(exc)) from exc


def compare(actual,expected,spec,*,input_text=None):
    """Shared PUB/offline pass predicate. Invalid participant output is False.

    A malformed expected output/input or unreviewed spec raises a distinct error;
    callers must stop/mark UNKNOWN instead of converting it to a model failure.
    """
    validate_spec(spec)
    if isinstance(actual,str) and '\ufffd' in actual:
        raise ComparatorObservationError('Replacement-decoded stdout cannot establish original testlib byte semantics; preserve UNKNOWN')
    if spec['kind']=='line_tokens':
        try:_bytes(expected)
        except (OutputSyntaxError,UnicodeError) as exc:raise ComparatorReferenceError(str(exc)) from exc
        try:return _compare_lines(actual,expected)
        except (OutputSyntaxError,UnicodeError):return False
    reference=reference_tokens(expected,spec,input_text=input_text)
    try:
        words=tokens(actual)
        if spec['kind']=='integer_tokens':words=[parse_int64(w) for w in words]
        elif spec['kind']=='yesno_tokens':
            words=[w.lower() for w in words]
            if any(w not in (b'yes',b'no') for w in words):return False
        return words==reference
    except (OutputSyntaxError,UnicodeError):return False


def comparator_identity(spec):
    validate_spec(spec);canonical=json.dumps(spec,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    return dict(name='LCBPro_static_reviewed_per_task_comparator_v1',kind=spec['kind'],task_id=spec['task_id'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),spec_sha256=hashlib.sha256(canonical).hexdigest(),
        checker_sha256=spec['checker_sha256'],testlib_sha256=TESTLIB_SHA256,testlib_revision=TESTLIB_REVISION,
        output_transport='Raw bytes, or existing executor UTF8 text; U+FFFD in text is unresolved and raises ComparatorObservationError',
        evidence='Static checker/header semantics + public-answer variants; no original checker execution or dynamic equivalence proof')


def execution_identity(sandbox,spec):
    identity=original.execution_identity(sandbox)
    identity['comparator']=comparator_identity(spec)
    return identity


def evaluate_suite(sandbox,code,tests,limits,visibility,*,spec):
    """Same isolated execution/60s CPU/72s wall/64MiB input for PUB and scoring.

    This function accepts supplied test records. It cannot load sealed
    packages or grant permission to evaluate heldout data. Callers control freeze.
    Bind execution_identity(sandbox,spec) into every cache key for both scopes.
    """
    validate_spec(spec)
    required=original.frozen_limits()|dict(input_bytes=64*1024**2)
    if limits!=required:raise ComparatorSpecError('All original execution limits plus64MiB input must be shared')
    if not isinstance(code,bytes) or not code or len(code)>limits['code_bytes'] or not tests:raise ValueError('Bounded original program/nonempty ordered suite required')
    if visibility not in {'public','dev_private','frozen_private'}:raise ValueError('Explicit evaluation visibility required')
    for row in tests:
        if set(row)!={'input','output'} or any(not isinstance(v,str) for v in row.values()) or len(row['input'].encode())>limits['input_bytes']:raise ValueError('Explicit input/output strings within64MiB required')
        # Validate reference before launching any candidate process.
        if spec['kind']!='line_tokens':reference_tokens(row['output'],spec,input_text=row['input'])
        else:_bytes(row['output'])
    start=time.perf_counter();cpu=time.thread_time();cases=[]
    with sandbox.lock,(sandbox.runtime_dir/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for index,row in enumerate(tests):
            result=sandbox._run_case(code.decode('utf-8'),row['input'],original.legacy_module().ExecutionLimits(**limits))
            if result['status']=='SANDBOX_SETUP_ERROR':raise RuntimeError('Isolation failed; no unsafe fallback')
            if result['status']=='EXECUTED':result['status']='PASS' if compare(result['stdout'],row['output'],spec,input_text=row['input']) else 'WRONG_ANSWER'
            cases.append(result|dict(test_index=index))
            if result['status']!='PASS':break
    observed=len(cases);passed=observed==len(tests) and all(r['status']=='PASS' for r in cases);child=sum(r['cpu_seconds'] for r in cases);host=time.thread_time()-cpu
    return dict(status='PASS' if passed else 'FAIL',all_passed=passed,total=len(tests),evaluated=observed,not_evaluated_after_failure=len(tests)-observed,
        cases=cases+[dict(test_index=i,status='NOT_EVALUATED_AFTER_FAILURE') for i in range(observed,len(tests))],wall_seconds=time.perf_counter()-start,
        child_wall_seconds=sum(r['wall_seconds'] for r in cases),cpu_seconds=child+host,child_cpu_seconds=child,host_thread_cpu_seconds=host,
        visibility=visibility,stop_on_failure=True,limits=limits,isolation=sandbox.runtime_manifest,code_sha256=hashlib.sha256(code).hexdigest(),comparator=comparator_identity(spec))
