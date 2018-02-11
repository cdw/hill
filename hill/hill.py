#!/usr/bin/env python

#          +-----+       +----+        
#        +-+ SEE +-+   +-+ CE +--+
#        | +-----+ |   | +----+  |
# Ft+----+         +---+         +--+Ft
#        | +-----+ |   | +-----+ |
#        +-+ SDE +-+   +-+ PEE +-+
#          +-----+       +-----+     

"""

If you are looking at this because you want to understand how a Hill-type 
model is written, this might be useful but there are several other references
that might be of use as well, including:
    - http://youngmok.com/hill-type-muscle-model-with-matlab-code/

"""

import numpy as np
import scipy.optimize


class BucklingSpring:
    """A spring that can bear tension but not compression"""
    def __init__(self, rest, k):
        self.rest = rest
        self.k = k

    def force(self, l):
        # allow stiffness to collapse if shorter than rest
        k = self.k if l>=self.rest else 0
        return k * (l - self.rest)


class PEE:
    """Parallel elastic element"""
    def __init__(self, ce):
        self.l_0  = 0.9*ce.l_opt # rest normalized to ce opt 
                                 # length (Guenther et al., 2007)
        self.v = 2.5  # exponent of F_PEE (Moerl et al., 2012)
        self.F = 2.0  # force of PEE if l_CE is stretched to 
                      # deltaWlimb_des (Moerl et al., 2012)
        self.K = self.F * (ce.F_max / 
                 (ce.l_opt*(ce.DeltaW_des + 1 - self.l_0) )**self.v ) 
                 # factor of non-linearity in F_PEE (Guenther et al., 2007)

    def force(self, length):
        """Force of the parallel elastic element"""
        if length >= self.l_0:
            F = self.K*(length-self.l_0)**(self.v)
        else: # shorter than slack length
            F = 0
        return F


class SEE:
    """Serial elastic element"""
    def __init__(self, l_0=0.172, DeltaF_0=568.0, 
                 DeltaU_nll=0.0425, DeltaU_l=0.017):
        """The series elastic element
        Values are derived from Gunther et al 2007 and  Haeufle et al 2014

        Parameters
        ----------
        l_0: float (0.172)
            rest l in [m] (Kistemaker et al., 2006)
        DeltaF_0: float (568.0)
            both force at the transition and force increase in the linear
            part in [N] (~ 40% of maximal isometric muscle force)
        DeltaU_nll: float (0.0425) 
            relative stretch at non-lin/lin transition (Moerl et al., 2012)
        DeltaU_l: float (0.017)
            relative additional stretch in the linear part providing a 
            force increase of deltaF_SEE0 (Moerl, 2012)
        """
        ## Save to structure
        self.l_0 = l_0
        self.DeltaF_0 =  DeltaF_0
        self.DeltaU_nll = DeltaU_nll 
        self.DeltaU_l = DeltaU_l 
        ## Calculate derived values 
        self._calculate_derived_values()

    def _calculate_derived_values(self):
        """Update the derived values, allows setting upstream parameters
        and reinitializing with new generative values
        """
        ## Get shorthand names
        l_0 = self.l_0
        DeltaU_nll = self.DeltaU_nll
        DeltaU_l = self.DeltaU_l
        DeltaF_0 = self.DeltaF_0
        ## Derived values
        l_nll = (1 + DeltaU_nll)*l_0    # length of non-linear region
        v     = DeltaU_nll / DeltaU_l   # exponent of non-lin region
        Knl   = DeltaF_0 / (DeltaU_nll*l_0)**v # spring in non-lin
        Kl    = DeltaF_0 / (DeltaU_l*l_0)      # spring in lin
        ## Save to structure
        self.l_nll, self.v = l_nll, v
        self.Kl, self.Knl =  Kl, Knl

    def force(self, length):
        """Force of the serial elastic element
        
        Parameters
        ----------
        length: float
            length of the serial elastic element
        """
        if (length>self.l_0) and (length<self.l_nll): # non-linear part
            F = self.Knl*((length-self.l_0)**(self.v))
        elif length>=self.l_nll: # linear part
            F = self.DeltaF_0+self.Kl*(length-self.l_nll)
        else: # slack length
            F = 0
        return F


class SDE:
    """Serial damping element"""
    def __init__(self, muscle):
        self.muscle = muscle
        self.D = 0.3 # dimensionless factor to scale d_max (Moerl et al., 2012)
        self.R = 0.01 # min D normalised to d_max (Moerl et al., 2012)
        ## read CE values to calculate max damp
        ce_F_max = muscle.ce.F_max
        ce_a_rel0 = muscle.ce.a_rel0
        ce_b_rel0 = muscle.ce.b_rel0
        ce_l_opt =  muscle.ce.l_opt
        ## Max D value in [Ns/m] (Moerl et al., 2012)
        self.D_max = self.D * (ce_F_max * ce_a_rel0) / (ce_l_opt * ce_b_rel0)

    def force(self, ce_length=None, q=None):
        ce_length = self.muscle.ce.l if ce_length is None else ce_length
        q = self.muscle.q if q is None else float(q)
        f_ce = self.muscle.ce.force(ce_length, q)
        f_max = self.muscle.ce.F_max
        f_pee = self.muscle.pee.force(ce_length)
        v_mus = self.muscle.v
        v_ce = self.muscle.ce.v
        f = (self.D_max * 
             ((1-self.R)*(f_ce + f_pee)/f_max + self.R) * 
             (v_mus - v_ce))
        return f


class CE:
    """Contractile element"""
    def __init__(self, muscle):
        """Contractile element
        F_max: max force of extensor (Kistemaker et al., 2006), Newtons
        l_opt: optimal length of extensor (Kistemaker et al., 2006), meters
        DeltaW_des: width of normalized bell curve in descending limb (Moerl et al., 2012)
        v_des: exponent for descending limb (Moerl et al., 2012)
        """
        self.l_init = -1 # to be overwritten by muscle
        self.l = self.l_init
        self.muscle = muscle
        self.F_max = 1420.0 
        self.l_opt = 0.092 
        self.DeltaW_des = 0.35  
        self.DeltaW_asc = 0.35
        self.v = 0.0 #TODO document, needs to be updated/changed
        self.v_des = 1.5
        self.v_asc = 3.0 
        self.a_rel0 = 0.25 # max a_rel (Guenther, 1997, S. 82)
        self.b_rel0 = 2.25 # max b_rel (Guenther, 1997, S. 82)

    def iso_force(self, length):
        """Isometric force (Force length relation) via Guenther et al. 2007
        Expressed as a normalized force (0,1)
        """
        if length >= self.l_opt: # descending limb
            DeltaW, v = self.DeltaW_des, self.v_des
        else: # ascending limb
            DeltaW, v = self.DeltaW_asc, self.v_asc
        F = np.exp(-(np.abs(((length / self.l_opt) - 1) / DeltaW))**v)
        return F

    def a_rel(self, length, q):
        if length < self.l_opt: #ascending limb
            a_len = 1.0
        else: 
            a_len = self.iso_force(length)
        a_q =  1.0/4.0*(1.0+3.0*q)
        a = self.a_rel0 * a_len * a_q
        return a

    def b_rel(self, length, q):
        b_len = 1.0
        b_q = 1.0/7.0*(3.0+4.0*q)
        b = self.b_rel0 * b_len * b_q
        return b

    def force(self, length=None, q=None):
        """ 
        Force during concentric contractions
        From Haeufle et al, 2014:
            $$ F_{ce,c}(v_{ce} \leq 0) = F_{max}  
            \left( \frac{q F_{iso} + a_{rel}}
            {1-\frac{v_{ce}}{b_{rel} l_{ce, opt}}}-a_rel \right) $$
        """
        length = self.l if length is None else float(length)
        q = self.muscle.q if q is None else float(q)
        a = self.a_rel(length, q)
        b = self.b_rel(length, q)
        F = self.F_max * ( 
            (q * self.iso_force(length) + a) / 
            (1 - self.v/(b*self.l_opt)) - a)
        return F

    def update_v(self, muscle_length, q):
        """Compute the velocity of the contractile element"""
        # Values of the CE
        f_isom = self.iso_force(self.l)
        f_max = self.F_max
        l_opt = self.l_opt
        a_rel = self.a_rel(muscle_length, q)
        b_rel = self.b_rel(muscle_length, q)
        # SEE
        see_length = muscle_length - self.l
        f_see = self.muscle.see.force(see_length)
        # PEE
        f_pee = self.muscle.pee.force(self.l)
        # SDE
        r_sde = self.muscle.sde.R
        d_max_sde = self.muscle.sde.D_max
        # Muscle
        v_mus = self.muscle.v
        # Coefficients 
        #d0 = l_opt * b_rel * d_max_sde * \
        #        (r_sde+(1-r_sde) * (q*f_isom+f_pee/f_max))
        #c2 = d_max_sde * (r_sde - (a_rel - f_pee/f_max)*(1-r_sde))
        #c1 = -(c2*v_mus+d0+f_see-f_pee+f_max*a_rel)
        #c0 = d0*v_mus+l_opt*b_rel*(f_see-f_pee-f_max*q*f_isom)
        # Velocity
        #v = (-c1-np.sqrt(c1*c1-4*c2*c0))/(2*c2)
        #self.v = v
        #return v
        # Calculate f_st (f_static + f_trans)
        f_st = 0
        ## Coefficients 
        Alpha = (d_max_sde * (f_max * (r_sde * (1 + a_rel) - a_rel) - 
                              (f_pee - f_st) * (r_sde - 1)))
        Beta = (b_rel * d_max_sde * l_opt * (f_pee - f_st) * (r_sde - 1) - 
                a_rel * f_max**2 + 
                f_max * (f_pee - f_see - f_st + 
                         b_rel * d_max_sde * l_opt * 
                         (q * f_isom * (r_sde - 1) - r_sde)) - 
                 Alpha * v_mus)
        Chi = (-b_rel * l_opt * 
               (f_max * (f_pee - f_see - f_st) + 
                d_max_sde * v_mus * (f_st + f_pee * (r_sde - 1) - (f_max + f_st) * r_sde) + 
                q * f_isom * f_max * (f_max + d_max_sde * v_mus * (r_sde - 1))))
        # Velocity
        v = (-Beta - np.sqrt(Beta**2 - 4 * Alpha * Chi)) / (2*Alpha)
        self.v = v
        return v
         
class Muscle:
    """ All the elements together
    Force balance is: $F_{ce} + F_{pee} = F_{see} + F_{sde}$ or, with more
    detail:
    TODO: FIX LATEX $$F_{ce}(l_{ce}, dot_l_ce, q) + F_pee(l_ce) = 
    F_see(l_cw, l_mtu) + F_sde(l_ce, v_ce, v_mtc, q)
    """
    def __init__(self, initial_length=0.264, initial_q=0, dt=0.001):
        """Default value for initial length is 0.092+0.172, the sum of the 
        CE optimal length and the SEE rest length.
        """
        # Set own values
        self.l_init = initial_length
        self.l = initial_length
        self.q_init = initial_q
        self.q = initial_q
        self.dt = dt # in seconds
        self.v = 0.0 # prescribed, muscle velocity starts at rest
        # Create elements
        self.ce = CE(self)
        self.pee = PEE(self.ce)
        self.see = SEE()
        self.sde = SDE(self)
        # Initialize element lengths
        self.ce.l_init = self.equilibrium_ce_length(self.q)
        self.ce.l = self.ce.l_init 

    def equilibrium_ce_length(self, q):
        """Run to figure out the equilibrium length of contractile element
        Finds the static force balance length between CE, PEE, and SEE 
        at current muscle length and given activation level
        """
        residual_force = lambda l: self.residual_force(l, self.l, q)
        ce_length_eq = scipy.optimize.fsolve(residual_force, 0.092)[0]
        return ce_length_eq

    def static_forces(self, ce_length=None, muscle_length=None, q=None):
        """Use to find initial lengths"""
        ## Parse passed values to stored if blank
        ce_length = self.ce.l if ce_length is None else ce_length
        muscle_length = self.l if muscle_length is None else muscle_length
        q = self.q if q is None else q
        ## Calc forces
        see_length = muscle_length - ce_length
        f_see = self.see.force(see_length)
        f_ce = q * self.ce.iso_force(ce_length) * self.ce.F_max
        f_pee = self.pee.force(ce_length)
        return f_see, f_ce, f_pee

    def residual_force(self, ce_length=None, muscle_length=None, q=None): 
        """Use to find initial lengths"""
        f_see, f_ce, f_pee = self.static_forces(ce_length, muscle_length, q)
        residual = f_see - f_ce - f_pee
        return residual

    def tension(self, ce_length=None, muscle_length=None, q=None):
        """The tension felt at the muscle ends"""
        f_see, f_ce, f_pee = self.static_forces(ce_length, muscle_length, q)
        force_sum = f_ce + f_pee
        return force_sum

    def step(self, q=None, dt=None):
        """Take a step, q is activation (0,1), dt is time in sec"""
        q = self.q if q is None else float(q)
        self.q = q
        dt = self.dt if dt is None else float(dt)
        self.dt = dt
        # Find CE speed
        self.ce.update_v(self.l, q)
        # Find element forces
        f_see = self.see.force(self.l - self.ce.l)
        f_sde = self.sde.force(self.ce.l, q)
        f_mtc = f_see + f_sde
        # TODO left off here. Need to validate the 1000 mass scaling factor for
        # conversion to a, not even sure we want to return a
        a_mtc = f_mtc / 1000.0 
        self.ce.l += self.ce.v*dt
        #return str(self.l_MTC) + " " + str(self.v_MTC) + " " + str(self.v_CE) + " " + str(self.l_CE) + " " + str(f_mtc) + " " + str(a_mtc)
        return a_mtc

    def log_dict(self):
        """Log the state of the model and return as a dict"""
        d = {"ce_f":self.ce.force(),
             "ce_l":self.ce.l,
             "ce_v":self.ce.v,
             "pee_f":self.pee.force(self.ce.l),
             "pee_l":self.ce.l,  # same length as ce
             "pee_v":self.ce.v,  # thus same v
             "see_f":self.see.force(self.l - self.ce.l),
             "see_l":self.l - self.ce.l,
             "see_v":self.v-self.ce.v,  # muscle vel minus ce vel
             "sde_f":self.sde.force(),
             "sde_l":self.l - self.ce.l,  # same as see
             "sde_v":self.v-self.ce.v}  # same as see
        return d
