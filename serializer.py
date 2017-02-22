def serialize(p, filename):
  with open(filename, 'w') as f:
    slices = [p.pslices_i[k] for k in p.pslices_i if p.pslices_i[k].score() > 0]
    f.write(str(len(slices)) + "\n")
    for s in slices:
      f.write("{} {} {} {}\n".format(s.x, s.xe-1, s.y, s.ye-1))
