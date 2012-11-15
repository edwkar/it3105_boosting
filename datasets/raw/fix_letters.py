with open('letter-recognition.data') as f:
  for line in f:
    line = line.strip()
    vals = line.split(',')
    print ','.join(vals[1:] + [str(ord(vals[0])-ord('A'))])
