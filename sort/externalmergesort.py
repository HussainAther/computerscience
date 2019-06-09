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

