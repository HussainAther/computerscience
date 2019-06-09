#include <bits/stdc++.h>
using namespace std;

struct MinHeapNode
{
    // The element to be stored
    int element;
    // index of the array from which the element is taken
    int i;
};

// Comparison object to be used to order the heap
struct comp
{
    bool operator()(const MinHeapNode lhs, const MinHeapNode rhs) const
    {
        return lhs.element > rhs.element;
    }
};

FILE* openFile(char* fileName, char* mode)
{
    FILE* fp = fopen(fileName, mode);
    if (fp == NULL)
    {
        perror("Error while opening the file.\n");
	exit(EXIT_FAILURE);
    }
    return fp;
}
 
// Merges k sorted files. Names of files are assumed to be 1, 2, ... k
void mergeFiles(char *output_file, int n, int k)
{
    FILE* in[k];
    for (int i = 0; i < k; i++)
    {
        char fileName[2];

        // convert i to string
	snprintf(fileName, sizeof(fileName), "%d", i);

	// Open output files in read mode.
	in[i] = openFile(fileName, "r");
	}
