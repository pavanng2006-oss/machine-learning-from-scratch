def count_comparisons_linear_search(arr, target):
    """
    arr:    a list of numbers
    target: the value being searched for

    Returns:
        (found, comparisons): found is True if target is in arr, False
        otherwise. comparisons is the number of element-to-target
        comparisons actually performed (stop as soon as a match is
        found -- don't keep scanning).
    """
    # TODO: Implement a left-to-right linear scan, counting comparisons.
    num_steps = 0
    found = False
    for i, val in enumerate(arr):
        num_steps += 1
        if val == target:
            found = True
            break

    return (found, num_steps)  
        

def count_comparisons_binary_search(sorted_arr, target):
    """
    sorted_arr: a list of numbers, already sorted ascending
    target:     the value being searched for

    Returns:
        (found, comparisons): same shape as count_comparisons_linear_search,
        but using binary search -- each comparison must actually halve
        the remaining search space.
    """
    # TODO: Implement binary search from Theory, counting one
    # comparison per midpoint check.
    found = False
    num_steps = 0
    low, high = 0, len(sorted_arr)
    while low < high:
        num_steps += 1
        mid = low + (high - low)//2

        if sorted_arr[mid] == target:
            found = True
            return (found, num_steps)

        elif sorted_arr[mid] < target:
            low = mid + 1
        else:
            high = mid
            
    return (found, num_steps)
