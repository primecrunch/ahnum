#!/bin/sh
# kite-prst against PRST 14.0.1888 reference output for 1203*2^90001-1: the proof point,
# product and certificate files must match PRST's byte for byte (compared by MD5), and the
# RES64s must agree. Usage: tests/prst_compat.sh build/kite-prst
set -u
kp=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
cd "$dir" || exit 1
fail=0
check() {  # file expected-md5
    got=$(md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1)
    if [ "$got" != "$2" ]; then echo "FAIL $1: $got, PRST has $2"; fail=1; fi
}
expect_line() {  # file pattern
    tr '\r' '\n' < "$1" | grep -q "$2" || { echo "FAIL $1: no line matching '$2'"; fail=1; }
}

"$kp" "1203*2^90001-1" -d -fermat a 3 -proof save 512 keep > save.out 2>&1 || fail=1
expect_line save.out "is not prime. RES64: 4BE9F1953E350085"
expect_line save.out "raw certificate RES64: EC81DC6893506908"
check prst_1107603830.proof.0 dafcb9cf37507263eaee0ece52991b3e
check prst_1107603830.proof.512 c57ae2618fb59c8652987543ab7cec29
check prst_1107603830.prod.0 57b767aa53d89f86a5fe742d66089abe
check prst_1107603830.prod.1 2d426d68ba20d327d66d58a1e0c33316
check prst_1107603830.prod.2 c3b3acd901f61818ad42b61337cadf9b
check prst_1107603830.prod.3 bfcbaed0176cd4a985744b04692d21f8
check prst_1107603830.prod.4 a58741b2b37270b7f213119e48d9ae4a
check prst_1107603830.prod.5 20d54b4ed93bd466f69c39a51a27113b
check prst_1107603830.prod.6 b8f2e1c7af03a5d277007f944335876e
check prst_1107603830.prod.7 1d8bcc840e8d7c3246b17898d1b96b34
check prst_1107603830.prod.8 3d38d7470434544fa95858a251bd3b8a

"$kp" "1203*2^90001-1" -d -fermat a 3 -proof build 512 > build.out 2>&1 || fail=1
expect_line build.out "certificate RES64: EC81DC6893506908"
check prst_1107603830.cert f3ae012634454e38aa7311deccccde56
[ -e prst_1107603830.proof.0 ] && { echo "FAIL build left point 0 behind without keep"; fail=1; }

mkdir v && cp prst_1107603830.cert v/verify.cert && cd v || exit 1
"$kp" "1203*2^90001-1" -d -proof cert verify.cert > cert.out 2>&1 || fail=1
expect_line cert.out "certificate RES64: EC81DC6893506908"
cd ..

"$kp" -v | grep -q "^kite version [0-9]*\.[0-9]*\.[0-9]*$" || { echo "FAIL -v banner"; fail=1; }

[ $fail = 0 ] && echo "prst compatibility: OK"
exit $fail
