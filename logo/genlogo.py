#!/opt/local/bin/python

import sys

sz = 720
R = 3/4

svg = """<svg version="1.1" xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     x="0" y="0" width="{sz}" height="{sz}"
     viewBox="0, 0, {sz}, {sz}">
  <defs>
    <style>
       @import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed&display=swap');
    </style>
  </defs>
  <g id="g1">
    <path d="M0,0 L{sz},0 L{sz},{sz} L0,{sz} L0,0 z" fill="#000000"/>
    <path d="M{sl},{sb} L{sm},{st} L{sr},{sb} z" fill="{sc}"/>
    <path d="M{ml},{mb} L{mm},{mt} L{mr},{mb} z" fill="{mc}"/>
    <path d="M{bl},{bb} L{bm},{bt} L{br},{bb} z" fill="{bc}"/>
    <text x="{tx}" y="{ty}">
      <tspan dx="{dx}" font-family="Barlow Semi Condensed" font-size="{fs}" fill="#FFFFFF">
        LA FORESTA
      </tspan>
    </text>
  </g>
</svg>"""

# Necessary for font import.
html = len(sys.argv) == 2 and sys.argv[1].endswith('-html')

class Tree:
    def __init__(self, w, h, x, y, c):
        self.l, self.m, self.r = round(x), round(x+w/2), round(x+w)
        self.b, self.t = round(y), round(y-h)
        self.c = c

fontsz = sz*41/216
margin_frac = 1/16
margin_top = sz * margin_frac
margin_left = sz * margin_frac
epsilon = margin_top/8
h_orig = sz*23/33

h = h_orig
b = Tree(h*R, h, margin_left, margin_top+h, "#789C22")
h = h*R
m = Tree(h*R, h, sz-margin_left-h*R, margin_top*2+h, "#789C22")
h = h*(R+1)/2
s = Tree(h*R, h, sz/2-h*R/2, margin_top+h+epsilon, "#97BF0F")

if html:
    print("<html>\n<body>\n<object>")
print(svg.format
      (sz=sz,
       sl=s.l,sm=s.m,sr=s.r,sb=s.b,st=s.t,sc=s.c,
       ml=m.l,mm=m.m,mr=m.r,mb=m.b,mt=m.t,mc=m.c,
       bl=b.l,bm=b.m,br=b.r,bb=b.b,bt=b.t,bc=b.c,
       fs=int(fontsz), dx=-int(fontsz/11),
       tx=round(margin_left),ty=round(margin_top+h_orig+fontsz),
       ))
if html:
    print("</object>\n</body>\n</html>")
