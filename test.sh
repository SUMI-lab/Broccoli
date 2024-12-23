echo "cp 1"
./broccoli.exe --env cp --depth 2 --num-iters 10000 --predicate-increment 0.1 0.1 0.1 0.1 --NeurIPS 0 1 --predicate-reasoning 1 > test.txt
echo "cp 2"
./broccoli.exe --env cp --depth 2 --num-iters 10000 --predicate-increment 0.1 0.1 0.1 0.1 --NeurIPS 0 1 --predicate-reasoning 1 > test2.txt