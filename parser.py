def parse(filename):
  with open(filename, 'r') as f:
    content = f.readlines()
    fline = content[0].split()
    x = int(fline[0])
    y = int(fline[1])
    _min = int(fline[2])
    _max = int(fline[3])
    pizza = [list(i.rstrip('\n')) for i in content[1:]]
    return [x, y, _min, _max, pizza]
