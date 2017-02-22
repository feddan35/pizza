def parse(filename):
  with open(filename, 'r') as f:
    content = f.readlines()
    fline = list(content[0])
    x = int(fline[0])
    y = int(fline[2])
    _min = int(fline[4])
    _max = int(fline[6])
    pizza = [list(i.rstrip('\n')) for i in content[1:]]
    return [x, y, _min, _max, pizza]
