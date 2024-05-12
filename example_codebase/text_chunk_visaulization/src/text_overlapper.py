def get_overlap_length(left: str, right: str):
    max_overlap = min(len(left), len(right))
    overlap = ""
    for i in range(1, max_overlap + 1):
        if left[-i:] == right[:i]:
            overlap = left[-i:]
    return len(overlap), overlap


def get_overlap_list(strings):
    overlaps = [get_overlap_length(strings[i], strings[i + 1]) for i in range(len(strings) - 1)]
    return overlaps

def unoverlap_list(strings):
    overlaps = get_overlap_list(strings)
    new_list = []
    for index, string in enumerate(strings):
        if index > 0 and overlaps[index - 1][0] > 0:
            new_list.append((overlaps[index - 1][1], True))

        left_overlap_length = 0 if index == 0 else overlaps[index - 1][0]
        right_overlap_length = 0 if index == len(strings) - 1 else overlaps[index][0]

        new_list.append(
            (string[left_overlap_length: len(string) - right_overlap_length], False)
        )
    return new_list

def main():
    strings = ["abcde", "defgh", "ghijkl"]
    unoverlapped = unoverlap_list(strings)
    print("Original strings:", strings)
    print("Overlapped String", get_overlap_list(strings))
    print("Unoverlapped list:", unoverlapped)


if __name__ == "__main__":
    main()
