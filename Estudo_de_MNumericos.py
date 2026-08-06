import numpy as np
import sympy as sp
import random as rand
import matplotlib.pyplot as plt
x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')
lab = sp.Symbol('lab')
f = ((x1-2)**4) + ((x1 - (2*x2))**2)
f_lam = sp.lambdify((x1,x2), f, 'numpy')

# Função erro (Calcula o erro percentual em cada iteração)

def c_erro(fa, fp):
    return np.linalg.norm(fp - fa)

# Funções em comum entre os Métodos

def fun(y):
    return f_lam(y[0],y[1])

def Grad(fun, p):
    gra = sp.Matrix([sp.diff(fun, x1), sp.diff(fun, x2)])
    Gradf = sp.lambdify((x1,x2), gra,'numpy')
    grad = np.array(Gradf(p[0], p[1]), dtype=float).flatten()
    return grad


# Métodos univariáveis (Método de Newton e Razão Áurea)

def ponto_univar(fb, lab):
    # Teste das derivadas
    dr1 = sp.diff(fb, lab)
    dr2 = sp.diff(dr1, lab)
    f1_num = sp.lambdify(lab, dr1, 'numpy')
    f2_num = sp.lambdify(lab, dr2, 'numpy')
    E = 0.001
    if dr1 is not None and dr2 is not None:
        x0 = 0.0
        max_iter = 60
        for _ in range(max_iter):
         d1 = f1_num(x0)
         d2 = f2_num(x0)
         if abs(d2) < 1e-12:
          break
         x1 = x0 - d1/d2
         if abs(x1 - x0) < E:
          return x1
         x0 = x1
         return x0
    else:
        fi = 0.618
        xh = 6
        xl = -6
        x1 = xh - fi*(xh - xl)
        x2 = xl + fi*(xh - xl)
        while abs(x1 - x2) > E:
        
          if sp.lambdify(x1, fb, 'numpy') < sp.lambdify(x2, fb, 'numpy'):
            xh = x2
          elif sp.lambdify(x1, fb, 'numpy') > sp.lambdify(x2, fb, 'numpy'):
            xl = x1
          else: 
            return (x1 + x2)/2
          x1 = xh - fi*(xh - xl)
          x2 = xl + fi*(xh - xl)
        return (x1 + x2)/2 
    
# Simplex de Nelder-Mead

def ordenar(vec):
    xi = vec[0]
    xj = vec[1]
    xk = vec[2]
    fxi = fun(xi)
    fxj = fun(xj)
    fxk = fun(xk)
    vec_fun = [fxi, fxj, fxk]
    xn_ini=np.array([])
    xs_ini=np.array([])
    xe_ini=np.array([])
    if min(vec_fun) == fxi:
        xn_ini=np.array(xi)
    elif min(vec_fun) == fxj:
        xn_ini=np.array(xj)
    else:
        xn_ini=np.array(xk)

    if max(vec_fun) == fxi:
        xe_ini =np.array(xi)
    elif max(vec_fun) == fxj:
        xe_ini=np.array(xj)
    else:
        xe_ini=np.array(xk)
    
    if not np.array_equal(xn_ini, xi) and not np.array_equal(xe_ini, xi):
        xs_ini=np.array(xi)
    elif not np.array_equal(xn_ini, xj) and not np.array_equal(xe_ini, xj):
        xs_ini=np.array(xj)
    else:
        xs_ini=np.array(xk)
    vec2 = [2,2]
    p0 = (xn_ini[0]+xs_ini[0])/vec2[0]
    p1 = (xn_ini[1]+xs_ini[1])/vec2[1]
    x_med = np.array([p0, p1]) 
    
    return xn_ini, xe_ini, xs_ini, x_med

def Transfor_simplex(vec, E):
    xn, xe, xs, c = ordenar(vec)
    alf = 1
    gama = [2,2]
    beta = [0.5,0.5]
    ro = [0.5,0.5]
    xr = [c[0] + (alf*(c[0]-xe[0])), c[1] + (alf*(c[1]-xe[1]))]
    xep = [c[0] + (gama[0]*(xr[0]-c[0])), c[1] + (gama[0]*(xr[1]-c[1]))]
    xc = [c[0] + (beta[0]*(xn[0] - c[0])), c[1] + (beta[1]*(xn[1] - c[1]))]
    
    while abs(xn[0]-xe[0])> E and abs(xn[1]-xe[1])>E:
     #Reflexão
     if fun(xs)>fun(xr)>=fun(xe):
         xn=xr
     #Expansão
     elif fun(xs)>fun(xe)>fun(xr):
         if fun(xe)>fun(xep):
             xn = xep
         else:
             xn = xe
     #Contração
     elif fun(xr)>fun(xs)>fun(xep):
         if fun(xc)<fun(xn):
             xn = xc
     #Contração encolhida
     else:
         xnl = [xe[0]+(ro[0]*(xn[0]-xe[0])),xe[1]+(ro[1]*(xn[1]-xe[1]))]
         xsl = [xe[0]+(ro[0]*(xs[0]-xe[0])),xe[1]+(ro[1]*(xs[1]-xe[1]))]
         xn = xnl
         xs = xsl
         
     
     vec_alsi =[xs, xe, xn]
     xn, xe, xs, c = ordenar(vec_alsi)
     xr = [c[0] + (alf*(c[0]-xn[0])),c[1] + (alf*(c[1]-xn[1]))]
     xep = [c[0] + (gama[0]*(xr[0]-c[0])), c[1] + (gama[1]*(xr[1]-c[1]))]
     xc = [c[0] + (beta[0]*(xn[0] - c[0])), c[1] + (beta[1]*(xn[1] - c[1]))]
     
        
    rsl =np.array([(xn[0]+xe[0])/2,(xn[1]+xe[1])/2])
    
    return rsl
    
# Método Fletcher-Reeve

def etapa(y, d, var):
    return [y[0]+var*d[0], y[1]+var*d[1]]

def Metodo_Fletcher_Reeves(f, y0, E, lab):
    normgrad = np.linalg.norm(np.array(Grad(f,y0), dtype=float))

    while normgrad > E:
     d0 = -1*(Grad(f,y0))
     etp = etapa(y0, d0, lab)
     fauxi = fun(etp)
     labaux = ponto_univar(fauxi, lab)
     y1 = [float(y0[0]+labaux*d0[0]), float(y0[1]+labaux*d0[1])]
     normagrad1=np.linalg.norm(np.array(Grad(f,y0), dtype=float))
     normagrad2=np.linalg.norm(np.array(Grad(f,y1), dtype=float))
     alfa = (normagrad2**2)/(normagrad1**2)
     d0 = -1*(Grad(f,y1))+(alfa*d0)
     etp = etapa(y1, d0, lab)
     fauxi = fun(etp)
     labaux = ponto_univar(fauxi, lab)
     y0 = [float(y1[0]+labaux*d0[0]), float(y1[1]+labaux*d0[1])]
     normgrad = np.linalg.norm(np.array(Grad(f,y0), dtype=float))
     

    return y0

# Método de Newton Multidimensional

def Matriz_H_num(fun, p):
    
    h = sp.Matrix([[sp.diff(sp.diff(fun, x1), x1), sp.diff(sp.diff(fun, x2), x1)],
                   [sp.diff(sp.diff(fun, x1), x2), sp.diff(sp.diff(fun, x2), x2)]])
    h_num = h.subs({x1:p[0], x2:p[1]})
    
    return h_num

def Metodo_Newton_Multivar(fun, p, E):
    grad_f = Grad(fun, p)
    H = Matriz_H_num(fun, p)
    H_inv = H.inv()
    x1 = p - (H_inv @ grad_f)
    norma = ((x1[0] - p[0])**2 + (x1[1] - p[1])**2)**0.5 
    
    while norma > E:
        p = x1
        grad_f = Grad(fun, p)
        H = Matriz_H_num(fun, p)
        H_inv = H.inv()
        x1 = p - (H_inv @ grad_f)
        norma = ((x1[0] - p[0])**2 + (x1[1] - p[1])**2)**0.5 
        
    return x1

def Random_Methods(y, E, f, max_inter):
 y = np.asarray(y, dtype=float)
 i_SPX = 0
 i_FR = 0
 i_NM = 0
 escolha = 0
 m_old = 0
 i = 0
 cerr = []
 while np.linalg.norm(y)> E and max_inter > i:
    opcoes = [1,2,3]
    if m_old in opcoes:
       opcoes.remove(m_old)
    escolha = rand.choice(opcoes)
    if escolha == 1:
       vec = [y,(0,0),(-1,-3)]
       x_resp = Transfor_simplex(vec, E)
       m_old = 1
       i_SPX +=1
    elif escolha == 2:
       x_resp = Metodo_Fletcher_Reeves(f, y, E, lab)
       m_old = 2
       i_FR += 1
    else:
       x_resp = Metodo_Newton_Multivar(f, y, E)
       m_old = 3
       i_NM += 1
    x = np.asarray(x_resp, dtype=float)
    ce = c_erro(y,x)
    cerr.append(ce)
    y = np.asarray(x_resp, dtype=float)
    i += 1

 return y, i_SPX, i_FR, i_NM, cerr

y_ini = [0, 3]
E = 0.0001
max_inter = 90
ymin, ctd_simplex, ctd_fletcher, ctd_newtonmult, v_err = Random_Methods(y_ini, E, f, max_inter)
print(f"Simplex:{ctd_simplex}\n")
print(f"Fletcher Reeve:{ctd_fletcher}\n")
print(f"Newton Multivariado:{ctd_newtonmult}\n")
print(f"O ponto mínimo é {ymin}")

x = np.linspace(1,max_inter,90)
plt.plot(x,v_err)
plt.grid(True)
plt.show()