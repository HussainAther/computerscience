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
    // FINAL OUTPUT FILE
    FILE *out = openFile(output_file, "w");

    // Create a min heap with k heap nodes.  Every heap node has first
    // element of scratch output file
    MinHeapNode harr[k];
    priority_queue<MinHeapNode, std::vector<MinHeapNode>, comp> pq;

    int i;
    for (i = 0; i < k; i++)
    {
        // break if no output file is empty and
        // index i will be no. of input files
        if (fscanf(in[i], "%d ", &harr[i].element) != 1)
            break;

        // Index of scratch output file
        harr[i].i = i;
        pq.push(harr[i]);
    }
    int count = 0;

    // Now one by one get the minimum element from min heap and replace
    // it with next element. Run till all filled input files reach EOF
    while (count != i)
    {
        // Get the minimum element and store it in output file
        MinHeapNode root = pq.top();
        pq.pop();
        fprintf(out, "%d ", root.element);

        // Find the next element that should replace current root of heap.
        // The next element belongs to same input file as the current
        // minimum element.
        if (fscanf(in[root.i], "%d ", &root.element) != 1 )
        {
            root.element = INT_MAX;
         count++;
        }

        // Replace root with next element of input file
        pq.push(root);
    }

    // close input and output files
    for (int i = 0; i < k; i++)
        fclose(in[i]);
    fclose(out);
    }


// Using a merge-sort algorithm, create the initial runs and divide them
// evenly among the output files
void createInitialRuns(char *input_file, int run_size, int num_ways)
{
    // For big input file
    FILE *in = openFile(input_file, "r");

    // output scratch files
    FILE* out[num_ways];
    char fileName[2];
    for (int i = 0; i < num_ways; i++)
    {
        // convert i to string
        snprintf(fileName, sizeof(fileName), "%d", i);

        // Open output files in write mode.
        out[i] = openFile(fileName, "w");
    }

    // allocate a dynamic array large enough to accommodate runs of
    // size run_size
    int* arr = (int*)malloc(run_size * sizeof(int));

    bool more_input = true;
    int next_output_file = 0;

    int i;
    while (more_input)
    {
        // write run_size elements into arr from input file
        for (i = 0; i < run_size; i++)
        {
            if (fscanf(in, "%d ", &arr[i]) != 1)
            {
                more_input = false;
                break;
            }
        }

        // sort array using merge sort
        sort(arr, arr + i);

        // write the records to the appropriate scratch output file
        // can't assume that the loop runs to run_size
        // since the last run's length may be less than run_size
        for (int j = 0; j < i; j++)
            fprintf(out[next_output_file], "%d ", arr[j]);

        next_output_file++;
    }

    // close input and output files
    for (int i = 0; i < num_ways; i++)
        fclose(out[i]);

    fclose(in);
}

/* Program to demonstrate External merge sort*/
int main()
{
    // No. of Partitions of input file
    int num_ways = 10;

    // The size of each partition
    int run_size = 1000;

    char input_file[] = "input.txt";
    char output_file[] = "output.txt";

    FILE* in = openFile(input_file, "w");

    srand(time(NULL));

    // generate input
    for (int i = 0; i < num_ways * run_size; i++)
        fprintf(in, "%d ", rand());

    fclose(in);

    // read the input file, create the initial runs,
    // and assign the runs to the scratch output files
    createInitialRuns(input_file, run_size, num_ways);

    // Merge the runs using the K-way merging
    mergeFiles(output_file, run_size, num_ways);

    return 0;
}
