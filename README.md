## Introduction
This is an implementation of applying the K-means algorithm on document clustering.

The main program is written in C++, while a python script `preprocess.py` is used to extract document data from the `Reuters-21578` dataset.

## Details of the K-means Algorithm
- Data objects are vectorized while each value of an element in the vector is the term frequency of the corresponding token (a word or a group of words) occurring in a document. Term frequency here means the times of a token occurring in a document.
- This program now only support cosine similarity, but can be easily extended by modifying the `_assignPoints()` function in the `lib/Kmeans.cpp` file. The cosine similarity is calculated via

    cos(x, c) = <x, c>/norm(c, 2)

where x is a vectorized data object, c is a centroid, `<.,.>` is the inner product of a object and a centroid, and `norm(., 2)` is the l2-norm of a centroid. For any vectorized data object, it has been normalized before participating in computation so that norm(x,2) == 1.
- Centroid is obtained as the mean of the corresponding normalized, vectorized data objects.
- This K-means algorithm calculates objective function via the dissimilarity and tries to minimize the objective function's value.
- This program can conduct clustering evaluation. It does not really evaluate the quality of the clustering solution it finds, but just shows the entropy and purity value of the clustering solution.
- This program use `mt19937` random engine to (pseudo-)randomly generate initial centroids.
- When empty clusters appear, a point would be pseudo-randomly picked from nonempty clusters as the cenroid of a new cluster who only contains the picked point.
- By default, the K-means algorithm stops iteration when no centroid changes. However, the KMeans class provides a function to set the threshold for this stop criterion.
- By default, the information generated during iteration would output into `std::clog`, this value can be changed in `main.cpp`.
- This program uses `Eigen3` to do vector/matrix computation.

## Preprocess of Data
By default, `preprocess.py` will extract data from `.sgm` files in the `reuters21578` folder. Due to that the clustering evaluation now does not support the fuzzy case, only those documents who only exactly have one topic would be extracted.

Then, by default, `preprocess.py` will generate the bag-of-words, 3-grams, 5-grams, 7-grams model based on words from those extracted documents and store these model into the `tokens_extracted` folder.

This script has been fully tested under Python 2.7 and 3.4. See the doc of `preprocess.py` for the details how it works.

Run the compiled program without any paramter to see help information, which introduces what parameters and structure of data files is required in details.

## How to run

Use

    make kmeans

to compile the program, while you can use

    make preprocess

to preprocess the data instead of runing `preprocess.py` by hand.

For Windows users, two compiled programs named `sphkmeans32.exe` and `sphkmeans64.exe` are provided.

For Mac OS users, a program named `sphkmeans.mac64` is provided, which is compiled under Mac OS 10.12.

For Linux users, you can compile the program yourself. If you use Ubuntun, two programs named `sphkmeans.lx32` and `sphkmeans.lx64` are provided. These two programs are compiled in Visual Studio 2015.

The program `sphkmeans` accept five parameters but not all of them  must be provided. Run it without any paramter to see the information of the parameters.

An automatical testing, bash script, `run.sh`, is provided. By default, it will automatically compile the program into `sphkmeans`, then extract tokens using `preprocess.py` from the `reuters21578` folder, then run a batch of clustering tests and put the best clustering solutions and all log information generated during clustering into the `log` folder. You may not want to run all the tests, since it may take about one hour to finish all tests.

## Finally

Contact me if you have any question.

Pei Xu, xuxx0884@umn.edu
