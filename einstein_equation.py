from sage.all import *
from sage.manifolds.manifold import Manifold
import numpy as np
from operator import methodcaller
from sage.ext.fast_callable import fast_callable

def hamiltonian_constraint(M, K, g):
    nab = g.connection()
    return g.ricci_scalar()-K.contract(0,1,K.up(g),0,1)+K.up(g,0).trace()**2

def momentum_constraint(M, K, g):
    nab = g.connection()
    return nab(K.up(g))["^ij_i"]-nab(K.up(g,0).trace()).up(g)

def dynamical_system(M, N, K, S, E, g):
    nab = g.connection()
    return -nab(nab(N))+\
           N*(g.ricci()+k*K-2*K.contract(K.up(g,0))+4*pi*((S-E)*g-2*S))

def Lflat(V,h,g):
    nab = h.connection()
    return 2*nab(V).up(g).symmetrize() - nab(V).trace() * 2*g.inverse() / 3

print("Defining metric")
M = Manifold(3,'M',structure="Riemannian")
C = M.chart("x y z")
x, y, z = C[:]
g = M.metric('g')
h = M.metric('h')
f = function('Psi')(x,y)
vex = function('v_x')(x,y)
vey = function('v_y')(x,y)
vez = function('v_z')(x,y)

g[0,0], g[1,1], g[2,2] = f**4*1, f**4*x**2, f**4*x**2*sin(y)**2
h[0,0], h[1,1], h[2,2] = 1, x**2, x**2*sin(y)**2

v = M.vector_field()
v[0] = vex
v[1] = vey
v[2] = vez
k = 0
K = 1/3*k*g+Lflat(v,h,g).down(g)/2/f**6
E, S = 0,0
N = M.scalar_field({C:1})

nr = 20
nt = 20
dr = 1
dt = pi/(nt-1)

print("Computing symbolic hamiltonian constraint")
mc = momentum_constraint(M,K,g)
print("Computing symbolic momentum constraint")
hc = hamiltonian_constraint(M, K, g)

def inner_scheme(p,wr,wt,wp,i,j):
    scheme = {}
    scheme[f] = p[i,j]
    scheme[vex(x=x,y=y)] = wr[i,j]
    scheme[vey(x=x,y=y)] = wt[i,j]
    scheme[vez(x=x,y=y)] = wp[i,j]

    scheme[diff(f,x)] = (p[i+1,j]-p[i-1,j])/2/dr
    scheme[diff(f,y)] = (p[i,j+1]-p[i,j-1])/2/dt
    scheme[diff(vex,x)] = (wr[i+1,j]-wr[i-1,j])/2/dr
    scheme[diff(vex,y)] = (wr[i,j+1]-wr[i,j-1])/2/dt
    scheme[diff(vey,x)] = (wt[i+1,j]-wt[i-1,j])/2/dr
    scheme[diff(vey,y)] = (wt[i,j+1]-wt[i,j-1])/2/dt
    scheme[diff(vez,x)] = (wp[i+1,j]-wp[i-1,j])/2/dr
    scheme[diff(vez,y)] = (wp[i,j+1]-wp[i,j-1])/2/dt

    scheme[diff(f,x,x)] = (p[i+1,j]-2*p[i,j]+p[i-1,j])/dr**2
    scheme[diff(f,y,y)] = (p[i,j+1]-2*p[i,j]+p[i,j-1])/dr**2
    scheme[diff(vex,x,x)] = (wr[i+1,j]-2*p[i,j]+wr[i-1,j])/dr**2
    scheme[diff(vex,y,y)] = (wr[i,j+1]-2*p[i,j]+wr[i,j-1])/dr**2
    scheme[diff(vey,x,x)] = (wt[i+1,j]-2*p[i,j]+wt[i-1,j])/dr**2
    scheme[diff(vey,y,y)] = (wt[i,j+1]-2*p[i,j]+wt[i,j-1])/dr**2
    scheme[diff(vez,x,x)] = (wp[i+1,j]-2*p[i,j]+wp[i-1,j])/dr**2
    scheme[diff(vez,y,y)] = (wp[i,j+1]-2*p[i,j]+wp[i,j-1])/dr**2

    scheme[diff(vex,x,y)] = (wr[i+1,j+1]-wr[i+1,j-1]-wr[i-1,j+1]+wr[i-1,j-1])/(4*dr*dt)
    scheme[diff(vey,x,y)] = (wt[i+1,j+1]-wt[i+1,j-1]-wt[i-1,j+1]+wt[i-1,j-1])/(4*dr*dt)
    scheme[diff(vez,x,y)] = (wp[i+1,j+1]-wp[i+1,j-1]-wp[i-1,j+1]+wp[i-1,j-1])/(4*dr*dt)
    return scheme

def border_scheme_outer(p,wr,wt,wp,i,j):
    scheme = {}
    scheme[f] = p[i,j]
    scheme[vex(x=x,y=y)] = wr[i,j]
    scheme[vey(x=x,y=y)] = wt[i,j]
    scheme[vez(x=x,y=y)] = wp[i,j]

    scheme[diff(f,x)] = (p[i,j]-p[i-1,j])/dr
    scheme[diff(f,y)] = (p[i,j+1]-p[i,j-1])/2/dt
    scheme[diff(vex,x)] = (wr[i,j]-wr[i-1,j])/dr
    scheme[diff(vex,y)] = (wr[i,j+1]-wr[i,j-1])/2/dt
    scheme[diff(vey,x)] = (wt[i,j]-wt[i-1,j])/dr
    scheme[diff(vey,y)] = (wt[i,j+1]-wt[i,j-1])/2/dt
    scheme[diff(vez,x)] = (wp[i,j]-wp[i-1,j])/dr
    scheme[diff(vez,y)] = (wp[i,j+1]-wp[i,j-1])/2/dt

    scheme[diff(f,y,y)] = (p[i,j+1]-2*p[i,j]+p[i,j-1])/dr**2
    scheme[diff(vex,y,y)] = (wr[i,j+1]-2*p[i,j]+wr[i,j-1])/dr**2
    scheme[diff(vey,y,y)] = (wt[i,j+1]-2*p[i,j]+wt[i,j-1])/dr**2
    scheme[diff(vez,y,y)] = (wp[i,j+1]-2*p[i,j]+wp[i,j-1])/dr**2

    return scheme

def border_scheme_inner(p,wr,wt,wp,i,j):
    scheme = {}
    scheme[f] = p[i,j]
    scheme[vex(x=x,y=y)] = wr[i,j]
    scheme[vey(x=x,y=y)] = wt[i,j]
    scheme[vez(x=x,y=y)] = wp[i,j]

    scheme[diff(f,x)] = (p[i+1,j]-p[i,j])/dr
    scheme[diff(f,y)] = (p[i,j+1]-p[i,j-1])/2/dt
    scheme[diff(vex,x)] = (wr[i+1,j]-wr[i,j])/dr
    scheme[diff(vex,y)] = (wr[i,j+1]-wr[i,j-1])/2/dt
    scheme[diff(vey,x)] = (wt[i+1,j]-wt[i,j])/dr
    scheme[diff(vey,y)] = (wt[i,j+1]-wt[i,j-1])/2/dt
    scheme[diff(vez,x)] = (wp[i+1,j]-wp[i,j])/dr
    scheme[diff(vez,y)] = (wp[i,j+1]-wp[i,j-1])/2/dt

    scheme[diff(f,y,y)] = (p[i,j+1]-2*p[i,j]+p[i,j-1])/dr**2
    scheme[diff(vex,y,y)] = (wr[i,j+1]-2*p[i,j]+wr[i,j-1])/dr**2
    scheme[diff(vey,y,y)] = (wt[i,j+1]-2*p[i,j]+wt[i,j-1])/dr**2
    scheme[diff(vez,y,y)] = (wp[i,j+1]-2*p[i,j]+wp[i,j-1])/dr**2

    return scheme


def border_scheme_up(p, wr, wt, wp, i, j):
    scheme = {}
    scheme[f] = p[i, j]
    scheme[vex(x=x, y=y)] = wr[i, j]
    scheme[vey(x=x, y=y)] = wt[i, j]
    scheme[vez(x=x, y=y)] = wp[i, j]

    scheme[diff(f, x)] = (p[i+1,j]-p[i-1,j])/ 2/ dr
    scheme[diff(f, y)] = (p[i,j+1]-p[i,j])/dt
    scheme[diff(vex, x)] = (wr[i+1,j]-wr[i-1,j])/2/dr
    scheme[diff(vex, y)] = (wr[i,j+1]-wr[i,j])/dt
    scheme[diff(vey, x)] = (wt[i+1,j]-wt[i-1,j])/2/dr
    scheme[diff(vey, y)] = (wt[i,j+1]-wt[i,j])/dt
    scheme[diff(vez, x)] = (wp[i+1,j]-wp[i-1,j])/2/dr
    scheme[diff(vez, y)] = (wp[i,j+1]-wp[i,j])/dt

    scheme[diff(f, y, y)] = (p[i,j+1]-2*p[i,j]+p[i,j-1])/dr**2
    scheme[diff(vex, y, y)] = (wr[i,j+1]-2*p[i,j]+wr[i,j-1])/dr**2
    scheme[diff(vey, y, y)] = (wt[i,j+1]-2*p[i,j]+wt[i,j-1])/dr**2
    scheme[diff(vez, y, y)] = (wp[i,j+1]-2*p[i,j]+wp[i,j-1])/dr**2

    return scheme


def border_scheme_down(p, wr, wt, wp, i, j):
    scheme = {}
    scheme[f] = p[i, j]
    scheme[vex(x=x, y=y)] = wr[i,j]
    scheme[vey(x=x, y=y)] = wt[i,j]
    scheme[vez(x=x, y=y)] = wp[i,j]

    scheme[diff(f, x)] = (p[i+1,j]-p[i-1,j])/2/dr
    scheme[diff(f, y)] = (p[i,j]-p[i,j-1])/dt
    scheme[diff(vex, x)] = (wr[i+1,j]-wr[i-1,j])/2/dr
    scheme[diff(vex, y)] = (wr[i,j]-wr[i,j-1])/dt
    scheme[diff(vey, x)] = (wt[i+1,j]-wt[i-1,j])/2/dr
    scheme[diff(vey, y)] = (wt[i,j]-wt[i,j-1])/dt
    scheme[diff(vez, x)] = (wp[i+1,j]-wp[i-1,j])/2/dr
    scheme[diff(vez, y)] = (wp[i,j]-wp[i,j-1])/dt

    scheme[diff(f, y, y)] = (p[i,j+1]-2*p[i,j]+p[i,j-1])/dr**2
    scheme[diff(vex, y, y)] = (wr[i,j+1]-2*p[i,j]+wr[i,j-1])/dr**2
    scheme[diff(vey, y, y)] = (wt[i,j+1]-2*p[i,j]+wt[i,j-1])/dr**2
    scheme[diff(vez, y, y)] = (wp[i,j+1]-2*p[i,j]+wp[i,j-1])/dr**2

    return scheme


def function_builder(eq):
    print("Preparing 4x{}x{} symbolic variables".format(nr,nt))
    p = np.array([[SR.var("p_{}_{}".format(i, j)) for i in range(nr)] for j in
                  range(nt)], dtype=object)
    wr = np.array([[SR.var("wr_{}_{}".format(i, j)) for i in range(nr)] for j in
                   range(nt)], dtype=object)
    wt = np.array([[SR.var("wt_{}_{}".format(i, j)) for i in range(nr)] for j in
                   range(nt)], dtype=object)
    wp = np.array([[SR.var("wp_{}_{}".format(i, j)) for i in range(nr)] for j in
                   range(nt)], dtype=object)

    fun = np.array([SR(0)] * (4 * nr * nt), dtype=object)

    print("Discretizing equations")
    for i in range(1, nr - 1):
        for j in range(1, nt - 1):
            scheme = inner_scheme(p, wr, wt, wp, i, j)
            fun[i * nt + j] = eq[0].subs(scheme)
            fun[400 + i * nt + j] = eq[1].subs(scheme)
            fun[800 + i * nt + j] = eq[2].subs(scheme)
            fun[1200 + i * nt + j] = eq[3].subs(scheme)
    for i in range(1, nt - 1):
        scheme = border_scheme_inner(p, wr, wt, wp, i, j)
        fun[i] = eq[4].subs(scheme)
        fun[400 + i] = eq[5].subs(scheme)
        fun[800 + i] = eq[6].subs(scheme)
        fun[1200 + i] = eq[7].subs(scheme)
    for i in range(1, nt - 1):
        scheme = border_scheme_outer(p, wr, wt, wp, i, j)
        fun[380 + i] = eq[8].subs(scheme)
        fun[780 + i] = eq[9].subs(scheme)
        fun[1180 + i] = eq[10].subs(scheme)
        fun[1580 + i] = eq[11].subs(scheme)
    for i in range(1, nr - 1):
        scheme = border_scheme_up(p, wr, wt, wp, i, j)
        fun[i * nt] = eq[12].subs(scheme)
        fun[400 + i * nt] = eq[13].subs(scheme)
        fun[800 + i * nt] = eq[14].subs(scheme)
        fun[1200 + i * nt] = eq[15].subs(scheme)
    for i in range(1, nr - 1):
        scheme = border_scheme_down(p, wr, wt, wp, i, j)
        fun[19 + i * nt] = eq[12].subs(scheme)
        fun[419 + i * nt] = eq[13].subs(scheme)
        fun[819 + i * nt] = eq[14].subs(scheme)
        fun[1219 + i * nt] = eq[15].subs(scheme)

    #     fun[0] = eq[8].subs(scheme)
    #     fun[19] = eq[8].subs(scheme)
    #     fun[399] = eq[8].subs(scheme)
    #     fun[380] = eq[8.subs(scheme)] puis increments de 400

    for i in range(4 * nr * nt):
        fun[i] = fun[i].subs({x: (i+1)*dr, y: j*dt})

    var = np.concatenate(
        (p.flatten(), wr.flatten(), wt.flatten(), wp.flatten()))
    print("Transforming function to fast callable")
    for i in range(4 * nr * nt):
        fun[i] = fast_callable(fun[i], vars=tuple(var), domain=float)
    return lambda x: map(methodcaller('__call__', *x), fun)

print("Declaring equations")
eq0 =(hc.expr()*f**5/8).expand()
eq1 =(mc[0]*f**10*x**2*sin(y)**2).expand().expr()
eq2 = (mc[1]*f**10*x**2*sin(y)**2).expand().expr()
eq3 = (mc[2]*f**10*x**2*sin(y)).expand().expr()
eq4 = (diff(f,x)+f/2/x)
eq5 = vex
eq6 = vey
eq7 = vez
eq8 = f
eq9 = vex
eq10 = vey
eq11 = vez
eq12 = diff(f,y)
eq13 = vex
eq14 = vey
eq15 = vez

print("building finite differences system")
fun = function_builder([eq0, eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10,
                        eq11, eq12, eq13, eq14, eq15])
print("done")












