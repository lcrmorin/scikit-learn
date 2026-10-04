StringDType performance and memory benchmark
==========================================

Install ``psutil`` in the environment containing the scikit-learn checkout and
NumPy version being tested. Run this script from that checkout::

    python benchmarks/bench_stringdtype.py --rows 1000 --repeats 1 \
        --output /tmp/stringdtype-smoke.json

For repeatable measurements, run one benchmark runner at a time on an otherwise
idle machine. Use independent environments for NumPy 2.2.4 and 2.5.3. Run a
representative matrix first::

    python benchmarks/bench_stringdtype.py --rows 100000 --repeats 3 \
        --profiles short long_tail --operations storage cache pipeline \
        --output /tmp/stringdtype-large.json

Expand to all operations/profiles, or use one million rows, when resources allow.
Fixed-width Unicode with a long outlier can require several GB at that scale.
The full matrix includes validation, unique/inverse/counts, ordinal and one-hot
encoding, label encoding/binarization, repeated target checks and accuracy scores,
unique-metadata caching, and an encoding/classification pipeline with held-out
prediction. LabelBinarizer uses sparse output to avoid quadratic dense storage.

The ``cache`` operation computes unique values once and requests them ten times
within one read-only metadata-cache scope. This measures reuse within an
operation; it does not retain results across separate public calls or mutations.

Every case compares identical values in StringDType, object and fixed-width
Unicode arrays. Category order, missing-value behavior, encoded values and
predictions are checked against an object-array control outside measured regions.
Unsupported missing-sentinel combinations are explicitly excluded. Failures are
written to JSON and cause a nonzero runner exit status. No timing threshold is a
unit-test assertion.

Memory interpretation
---------------------
Each case/repetition uses a new subprocess. Imports happen before the baseline
RSS snapshot. Construction timing excludes source-list generation; construction
peak allocations are included in process high-water RSS. Source lists are then
released before recording input RSS. RSS includes allocator retention, so its
change is not an exact accounting of owned string bytes.

Operation peaks are sampled at 1 ms intervals and may miss short peaks. The
process high-water RSS is also recorded on Unix, before allocating correctness
controls. It includes input construction, unlike the operation-only sampler.
``buffer_bytes`` is diagnostic only: neither StringDType nor object-array nbytes
accounts for all string payload storage. Do not interpret it as total memory.

Object inputs share repeated string objects by default, as can happen when
constructing arrays from a vocabulary. Repeat with ``--object-layout fresh`` to
measure independently allocated equal Python strings. Both are legitimate
workloads; neither should be silently substituted for the other.

Comparison and reporting
------------------------
Use medians across fresh-process repetitions for each operation/profile/dtype.
Report construction time, operation time, input RSS, operation peak RSS and
process high-water RSS separately. Keep sparse encoded output fixed across
representations. Do not mix dataset sizes, missing sentinels, object ownership,
or sparse/dense settings within a comparison.

To evaluate patch overhead, run the same script with environments importing the
base and patched scikit-learn checkouts, comparing object and Unicode paths.
StringDType operations that fail before the patch represent newly supported
behavior, not measurable speedups. The JSON records the imported checkout path,
commit, package versions, platform and dataset configuration. Timing runs on two
NumPy versions alone do not establish the patch's performance overhead.
