use POSIX;
use List::Util qw[min max];

$datetime = strftime "%Y-%m-%d-%H-%M-%S", localtime time;
$path = ">> ./output_".$datetime.".txt";
open(WF, $path) or die;

`g++ softmax_naive.cpp -o softmax_naive -fopenmp -O2`;
`g++ softmax_omp.cpp -o softmax_omp -fopenmp -O2`;
# `g++ softmax_omp_div.cpp -o softmax_omp_div -fopenmp -O2`;
`g++ softmax_mkl.cpp -o softmax_mkl -L/opt/intel/oneapi/mkl/latest/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -DMKL_ILP64 -m64 -I/opt/intel/oneapi/mkl/latest/include -fopenmp -O2 -march=native`;
`g++ softmax_mkl_avx.cpp -o softmax_mkl_avx -L/opt/intel/oneapi/mkl/latest/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -DMKL_ILP64 -m64 -I/opt/intel/oneapi/mkl/latest/include -fopenmp -O2 -g -march=native -mavx512vl`;


$time = 0.;
for ($i = 0; $i < 3; ++$i) {
    $tmp = `./softmax_naive`;
    $time += $tmp;
}
$time /= 3;
print WF "softmax_naive: ", $time, "\n";
print "ready softmax_naive\n";

$time = 0.;
for ($i = 0; $i < 3; ++$i) {
    $tmp = `./softmax_omp`;
    $time += $tmp;
}
$time /= 3;
print WF "softmax_omp: ", $time, "\n";
print "ready softmax_omp\n";

$time = 0.;
for ($i = 0; $i < 3; ++$i) {
    $tmp = `./softmax_omp_div`;
    $time += $tmp;
}
$time /= 3;
print WF "softmax_omp_div: ", $time, "\n";
print "ready softmax_omp_div\n";

$time = 0.;
for ($i = 0; $i < 3; ++$i) {
    $tmp = `./softmax_mkl`;
    $time += $tmp;
}
$time /= 3;
print WF "softmax_mkl: ", $time, "\n";
print "ready softmax_mkl\n";

$time = 0.;
for ($i = 0; $i < 3; ++$i) {
    $tmp = `./softmax_mkl_avx`;
    $time += $tmp;
}
$time /= 3;
print WF "softmax_mkl_avx: ", $time, "\n";
print "ready softmax_mkl_avx\n";