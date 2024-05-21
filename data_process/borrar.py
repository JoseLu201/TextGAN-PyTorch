def count_chars(filename):
    char_map = {}
    with open(filename, 'r') as file:
        for line in file:
            for char in line:
                if char in char_map:
                    char_map[char] += 1
                else:
                    char_map[char] = 1
    return char_map


# print(count_chars('partidos_v3/vox/orig_vox_tweets.txt').keys())
print(sorted(list(count_chars('partidos_v3/vox/orig_vox_tweets.txt').keys())))

print(len(count_chars('partidos_v3/vox/orig_vox_tweets.txt')))
print()
print(sorted(list(count_chars('partidos_v3/vox/vox_tweets.txt').keys())))
print(len(count_chars('partidos_v3/vox/vox_tweets.txt')))

def diff_keys(file1, file2):
    map1 = count_chars(file1)
    map2 = count_chars(file2)

    keys1 = set(map1.keys())
    keys2 = set(map2.keys())

    return keys1 - keys2

file1 = 'partidos_v3/vox/orig_vox_tweets.txt'
file2 = 'partidos_v3/vox/vox_tweets.txt'

print(diff_keys(file1, file2))