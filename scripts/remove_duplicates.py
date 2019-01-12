import sys

fin = open(sys.argv[1])
fout = open(sys.argv[2],"w")

for l in fin:
  w = l.strip().split()
  prev = ""
  newl = ""
  for word in w:
    if word == prev:
      continue
    newl += word + " "
    prev = word
  fout.write(newl+"\n")
