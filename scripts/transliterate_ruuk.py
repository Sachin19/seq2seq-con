from translitua import RussianSimple, translit

import sys

with open(sys.argv[1]) as fru, open(f"{sys.argv[1]}.romanized", "w") as fruroman:
    for l in fru:
        fruroman.write(translit(l, RussianSimple))

with open(sys.argv[2]) as fuk, open(f"{sys.argv[2]}.romanized", "w") as fukroman:
    for l in fuk:
        fukroman.write(translit(l))

print("done")