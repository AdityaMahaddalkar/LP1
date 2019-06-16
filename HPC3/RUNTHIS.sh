echo "\nCompiling Source Files"
g++ bubble_sort.cpp -fopenmp -o bsort
g++ merge_sort.cpp -fopenmp -o msort
clear
echo "\nCompilation successful"
sleep 2s
echo "\nRunning Bubble Sort"
./bsort
sleep 7s
clear
echo "\nRunning Merge Sort"
./msort
sleep 7s
clear
echo "\nExiting"
