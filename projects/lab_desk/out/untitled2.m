syms x m omega hbar positive
syms n integer

alpha = m*omega/hbar;
psi0 = exp(-alpha*x^2/2);

Hpsi0 = -hbar^2/(2*m)*diff(psi0,x,2) ...
        + (1/2)*m*omega^2*x^2*psi0;

E0 = simplify(Hpsi0/psi0)

En = hbar*omega*(n + 1/2)
DeltaE = simplify(subs(En,n,n+1) - En)