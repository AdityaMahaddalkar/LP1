echo "\n\nCompiling"
sleep 2s
g++ add_vectors.cpp -o add_vectors
g++ mul_vec_matrix.cpp -o mul_vec_matrix
g++ mul_matrix.cpp -o mul_matrix
clear
echo "\n\nExecuting addition of vectors"
sleep 2s
./add_vectors
sleep 7s
clear
echo "\n\nExecuting multiplication of vector and matrix"
sleep 2s
./mul_vec_matrix
sleep 7s
clear
echo "\n\nExecuting multiplication of matrices"
sleep 2s
./mul_matrix
sleep 7s
clear
echo "\n\nExiting"
