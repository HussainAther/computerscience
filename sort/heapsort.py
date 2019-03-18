"""
Heap sort is a comparison based sorting technique based on binary heap
data structure. It is similar to selection sort where we first find the
maximum element and place the maximum element at the end. We repeat the
same process for remaining element.
"""

def heapify(arr, n, i):
    max1 = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i] < arr[l]:
        max1 = l
    if r < n and arr[max1] < arr[r]:
        max1 = r
    if max1 != i:
        arr[i],arr[max1] = arr[max1],arr[i] # swap
        heapify(arr, n, max1)


def heapSort(a):
    n = len(a)
    for i in range(n, -1, -1):
        heapify(a, n, i)
    for i in range(n-1, 0, -1):
        a[i], a[0] = a[0], a[i]
        heapify(a, i, 0)
