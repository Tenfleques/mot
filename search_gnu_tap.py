lst = ["--with-wxmac","--with-cairo","--with-pdflib-lite","--with-x11","--without-lua"]

with open("gnuplot.json") as f:
    lines = f.readlines()
    for l in lines:
        contained = True
        for i in lst:
            if i not in l:
                contained = False
        if contained:
            print(l)