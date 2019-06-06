# -*- coding: utf-8 -*-

from __future__ import division, print_function
import sys
import numpy as np

from copy import deepcopy
from emcee.state import State

__all__ = ["PSOMove"]


class PSOMove(object):
    """
    This is a modified charged accelerated PSO
    """
    def __init__(self,tol_fit=1e-5,tol_norm=1e-1,distance_balance=1e-8):
        '''
        Arguments:
        tol_norm:             Relative tolerance of mean norm
        distance_balance:     Distance at which repulsive and attracting forces are equal
        '''

        #----- Convergence criteria ----------------
        self.tol_norm       = tol_norm
        self.tol_fit        = tol_fit 
        #-------------------------------------

        #----- This is PSO of Clerc and Kennedy 2002 
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.w  = 0.7298
        #--------------------------------------------

        #------------------------ Modified accelerated PSO of Blackwell Bently  -------------------------------------------------
        self.distance_core    = 1e-50  # Min relative distance at which accelerations turns to zero to avoid infinities.
        self.distance_balance = distance_balance   
        
        # Force inversely proportional to the distance
        self.c3 = 0.5*(self.c1+self.c2)*(self.distance_balance**2)  # constant to balance forces, independent in each dimension.
        # It is complemented in each dimension multiplying it by p0[i]


    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance
        Args:
            model:
            state: Either PSO state or emcee state
        """
        # Check that it is a pso state.
        state = pso_State(state)

        # Get the move-specific proposal.
        new_state = self.get_proposal(state, model.random)

        # Compute the lnprobs of the proposed position.
        new_log_prob, new_blobs = model.compute_log_prob_fn(new_state.coords)

        #------- Updates log and blobs ------
        new_state.log_prob = new_log_prob
        new_state.blobs    = new_blobs
        #---------------------------------

        # Accepted walkers------------
        accepted = np.isfinite(new_log_prob)

        # Update the state
        state = self.update(state,new_state, accepted)

        #----- Update the norm and convergence -----
        state.norm = self.state_norm(state)
        state.converged = self.converged(state)

        return state, accepted

    def get_proposal(self, state, random):
        new_state = pso_State(state,copy=True)

        cognitive_velocity = np.random.uniform(0,1,size=state.coords.shape) * (state.pbest_coords - state.coords)
        social_velocity    = np.random.uniform(0,1,size=state.coords.shape) * (state.gbest_coords - state.coords)

        new_state.velocities = (self.w * state.velocities) + (self.c1 * cognitive_velocity) + (self.c2 * social_velocity)

        new_state.coords     = new_state.coords + new_state.velocities + new_state.accelerations

        #------------- Computes acceleration -----------------------------------------
        for i,position in enumerate(new_state.coords):

            #---------- Relative distances -------------------------
            rho  = np.zeros_like(new_state.coords)
            for j,sibling in enumerate(new_state.coords):
                rho[j] = (sibling/position)-1.0
            #--------------------------------------------------------

            #-------- Valid particles for acceleration ---------------
            valid    = np.where((np.abs(rho) > self.distance_core))[0]
            #---------------------------------------------------------

            # To balance the forces, the constant c3 must be multiplied by position**3.
            # but since distance = rho * position the term multiplying c3 is just position instead of position**3

            #Force inversely proportional to distance
            accs        = np.zeros_like(rho)
            accs[valid] = -0.5*np.sign(rho[valid])*((np.abs(position))*self.c3)/(rho[valid])

            #---- In each dimension the resultant acceleration is addition over particles ---
            new_state.accelerations[i]   = np.sum(accs,axis=0)
        #-------------------------------------------------------------------------------------

        return new_state
        

    def update(self,state,new_state,accepted):
        """Update the ensemble with an accepted proposal.
        The particles that were not accepted stop at current position
        """
        idx_rejected = np.where(np.logical_not(accepted))[0]

        if len(idx_rejected != 0) :
            # print("The following particles were stopped")
            # print(idx_rejected)

            zeros_rejected = np.zeros((len(idx_rejected),new_state.velocities.shape[1]))

            new_state.coords[idx_rejected] = state.coords[idx_rejected]

            new_state.velocities[idx_rejected] = zeros_rejected
            new_state.velocities[idx_rejected] = zeros_rejected

            new_state.accelerations[idx_rejected] = zeros_rejected
            new_state.accelerations[idx_rejected] = zeros_rejected

        return new_state

    def converged(self,state):
        converged_norm = state.norm < self.tol_norm
        #----------- Fit ----------------------------------
        diffs = (state.log_prob-state.gbest_logprob)/state.gbest_logprob
        diffs = diffs[np.isfinite(diffs)]
        norm_fit  = np.mean(list(map(np.linalg.norm, diffs)))
        converged_fit  = norm_fit < self.tol_fit
        return converged_norm & converged_fit
    
    def state_norm(self,state):
        diffs = (state.coords-state.gbest_coords)/state.gbest_coords
        idx_valid = np.where(np.abs(state.gbest_coords) > self.tol_norm)[0]
        diffs = diffs[:,idx_valid]
        norm  = np.mean(list(map(np.linalg.norm, diffs)))
        return norm

    

class pso_State(State):
    """
    The state of the pso ensemble it is similar to the emcee state but with velocity and acceleration
    """

    # __slots__ = "velocities","accelerations","converged"

    def __init__(self, state,velocities=None,accelerations=None, converged=False,norm=None,copy=False,**kwargs):

        dc = deepcopy if copy else lambda x: x

        if not hasattr(state, "velocities"):
            self.velocities    = np.zeros_like(state.coords)
            self.accelerations = np.zeros_like(state.coords)
            #--------- Local best -----------------
            self.pbest_coords  = dc(state.coords)
            self.pbest_logprob = dc(state.log_prob)
            #--------- Global best -----------------------
            idx_gbest          = np.argmax(state.log_prob)
            self.gbest_coords  = dc(state.coords[idx_gbest])
            self.gbest_logprob = dc(state.log_prob[idx_gbest])
            #---------------------------------------------
            self.converged     = converged
            self.norm          = norm
            super(pso_State, self).__init__(state,copy,**kwargs)
            return

        self.velocities    = dc(np.atleast_2d(state.velocities))
        self.accelerations = dc(np.atleast_2d(state.accelerations))
        #------------ Local best ---------------------
        self.pbest_coords  = dc(state.pbest_coords)
        self.pbest_logprob = dc(state.pbest_logprob)
        #------------ Global best --------------------
        self.gbest_logprob = dc(state.gbest_logprob)
        self.gbest_coords  = dc(state.gbest_coords)
        #-----------------------------------------
        self.converged     = dc(state.converged)
        self.norm          = dc(state.norm)

        super(pso_State, self).__init__(state,copy,**kwargs)

        #=========== Update global and local bests ============

        #------------ Global best ----------------------
        idx_gbest     = np.argmax(state.log_prob)

        if state.log_prob[idx_gbest] > self.gbest_logprob :
            self.gbest_logprob = state.log_prob[idx_gbest]
            self.gbest_coords  = state.coords[idx_gbest]
        #---------------------------------------------------

        #------------- Local best -------------------------------------
        idx_update  = np.where(state.log_prob > self.pbest_logprob)[0]

        if len(idx_update) != 0:
            self.pbest_coords[idx_update]  = dc(state.coords[idx_update])
            self.pbest_logprob[idx_update] = dc(state.log_prob[idx_update])
        #------------------------------------------------------------------
        #====================================================================

    def __repr__(self):
        return "State(coords={0}, velocities={1}, accelerations={2}, log_prob={3}, blobs={4}, random_state={5},converged={6},pbest_logprob={7})".format(
            self.coords, self.velocities,self.accelerations,self.log_prob, self.blobs, self.random_state,self.converged,self.pbest_logprob
        )

    def __iter__(self):
        if self.blobs is None:
            return iter((self.coords, self.log_prob, self.random_state))
        return iter((self.coords, self.log_prob, self.random_state,
                     self.blobs))