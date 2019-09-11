#!/usr/bin/env python
# coding: utf-8

# # Assignment 2 - codes with explanations

# -------------------------------------------------------------------------------------------------
# #Q2)
# 
# state-value function corresponding to the given equiprobable policy, was found by solving linear system of bellman equations

# In[4]:


import numpy as np

#################setup

#actions are in this order: Up, Down, Left, Right

rewards=[0,-1,10,5] #possible rewards at each state

p=[] #p[present state][actions][next state][rewards] 
for i in range(25):#s(0-24)
    hold=[]
    for j in range(4):#a(nswe)
        l=[]
        for k in range(25):#s'(0-24)
            m=[0 for o in range(4)] #(0,-1,10,5)
            l.append(m)            
        hold.append(l)
    p.append(hold)
    
        
for i in range(len(p)):
    for j in range(len(p[i])):
        if(i==3):
            p[i][j][13][3]=1
        elif(i==1):
            p[i][j][21][2]=1
        else:    
            if(j==0):
                if(i-5<0):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i-5][0]=1
            elif(j==1):
                if(i+5>24):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i+5][0]=1
            elif(j==2):
                if(i==0 or i==5 or i==10 or i==15 or i==20):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i-1][0]=1   
            elif(j==3):
                if(i==4 or i==9 or i==14 or i==19 or i==24):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i+1][0]=1
                
        
        
#just verification of the MDP setup
for i in range(len(p)):
    for j in range(len(p[i])):
        su=0;count=1;
        for k in range(len(p[i][j])):
            for o in range(len(p[i][j][k])):
#                 print(k,o,p[i][j][k][o],count);count+=1
                su+=p[i][j][k][o]
#         if(su!=1):
#             print(i,j,su)
        assert(su==1)
        

#policy - equiprobable
policy=[] #policy[present state][actions]
for i in range(25):#present states
    given_State=[]
    for i in range(4):#actions
        given_State.append(0.25)
    policy.append(given_State)

gamma=0.9 #discounting
    
####################solving system of linear equations for bellman equations. Ax=b, x=state_value
b=[0 for i in range(25)]

for i in range(25):#present state
    for j in range(4):#actions
        for k in range(25):#next state
            for o in range(4):#rewards
                b[i]-=(rewards[o]*p[i][j][k][o]*policy[i][j])

A=[]

for i in range(25):#present state
    given_coeff=[]
    for j in range(25):#next state
        su=0
        for k in range(4):#actions
            for o in range(4):#rewards
                su+=(gamma*p[i][k][j][o]*policy[i][k])
        given_coeff.append(su)
        if(j==i):#adjusting the coeffs of the states that are on the LHS
            given_coeff[-1]-=1
    A.append(given_coeff)

        
A=np.array(A);b=np.array(b)
state_value=np.linalg.solve(A, b) #linear algebra solved using numpy

print(state_value)


# ------------------------------------------------------------------------------------------------
# #Q4)
# 
# This section uses policy iteration to solve system of non-linear equations. Policy Iteration has two functions:
# 
# 1>pol_eval():
# 
# INPUT: p,policy,state_value,gamma,epsilon,rewards
# p - MDP setup
# policy - current policy
# state_value - current state_values
# gamma - discounting
# epsilon - error margin
# rewards - list of possible rewards at each state
# 
# OUTPUT: calls pol_improve()
# 
# 2>pol_improve():
# 
# INPUT: same as pol_eval()
# 
# OUTPUT: calls pol_eval() or returns the final policy and the corresponding final state_values 

# In[5]:


import numpy as np
import math
import random
import copy

#######################setup

#actions are in this order: Up, Down, Left, Right

rewards=[0,-1,10,5] #possible rewards at each state

p=[] #p[present state][actions][next state][rewards]
for i in range(25):#s(0-24)
    hold=[]
    for j in range(4):#a(nswe)
        l=[]
        for k in range(25):#s'(0-24)
            m=[0 for o in range(4)] #(0,-1,10,5)
            l.append(m)            
        hold.append(l)
    p.append(hold)
    
        
for i in range(len(p)):
    for j in range(len(p[i])):
        if(i==3):
            p[i][j][13][3]=1
        elif(i==1):
            p[i][j][21][2]=1
        else:    
            if(j==0):
                if(i-5<0):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i-5][0]=1
            elif(j==1):
                if(i+5>24):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i+5][0]=1
            elif(j==2):
                if(i==0 or i==5 or i==10 or i==15 or i==20):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i-1][0]=1   
            elif(j==3):
                if(i==4 or i==9 or i==14 or i==19 or i==24):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i+1][0]=1
                
        
        
#just verification of the MDP setup
for i in range(len(p)):
    for j in range(len(p[i])):
        su=0;count=1;
        for k in range(len(p[i][j])):
            for o in range(len(p[i][j][k])):
#                 print(k,o,p[i][j][k][o],count);count+=1
                su+=p[i][j][k][o]
#         if(su!=1):
#             print(i,j,su)
        assert(su==1)
        

#policy initialization - equiprobable
policy=[] #policy[present state][actions] 
for i in range(25):#present states
    given_State=[]
    for i in range(4):#actions
        given_State.append(0.25)
    policy.append(given_State)

gamma=0.9 #discounting
    
###########################solving the system using policy iteration
        
epsilon=0.00001 #convergence tolerance
    
state_value=[random.random() for i in range(25)]#initialization
    

#policy evaluation
def pol_eval(p,policy,state_value,gamma,epsilon,rewards): #make these globals, function parameters
    while(True):
        
        delta=-math.inf #delta for convergence checking
        for i in range(25):#present state
            old=state_value[i]

            su=0
            for o in range(4):#actions
                for j in range(25):#next states
                    for k in range(4):#rewards
                        su+=(rewards[k]+(gamma*state_value[j]))*p[i][o][j][k]*policy[i][o]
            
            state_value[i]=su #state value of state i update in place
            
            delta=max(delta,abs(old-state_value[i]))
        
        if(delta<epsilon):
            break
        
    return pol_improve(p,policy,state_value,gamma,epsilon,rewards)
        
#policy improvement
def pol_improve(p,policy,state_value,gamma,epsilon,rewards):
    old_pol=copy.deepcopy(policy)
    
    for i in range(25):#present state
        ac_values=[0 for i in range(4)] #action values calculated
        for j in range(4):#actions
            for k in range(25):#next state
                for o in range(4):#rewards
                    ac_values[j]+=p[i][j][k][o]*(rewards[o]+(gamma*state_value[k]))

        #only all optimal actions assigned equal non-zero probabilities
        act=np.argmax(ac_values)
        for j in range(4):#actions
            policy[i][j]=0
            if(ac_values[act]==ac_values[j]):
                policy[i][j]=1

        su=sum(policy[i])
        for j in range(4):#actions
            policy[i][j]/=su
    
    if(policy!=old_pol):
        return pol_eval(p,policy,state_value,gamma,epsilon,rewards)
    else:
        return [policy, state_value] 
        

        
[new_policy, new_state_value]=pol_eval(p,policy,state_value,gamma,epsilon,rewards);
        
print();print("final policy");print(new_policy);print();print("final state values");print(new_state_value)
        
        
        
        
        
        
        


# ---------------------------------------------------------------------------------------------
# #Q6)both the iterations
# 
# policy iteration code is almost same as that in Q4, just the bug has been fixed here
# 
# Bug in the policy iteration: it may keep infintely switching between multiple optimal policies and hence never terminate(this was the policy iteration implemented in Q4). But the optimal state_values corresponding to any opitmal policy will be same. So we can just compare and check whether we have arrived at the optimal state_values or not(by checking whether the state_values have changed or not), and if yes then terminate the code from policy evaluation step itself.
# 
# For Value iteration: function is val_loop(): Its input and output are same as that for the functions of policy iteration
# 
# 
# Logging: an "iteration" variable introduced to properly log the policy and state_values at each iteration.
# Also the final policy and state_values are also shown.
# 
# 
# 

# In[8]:


import numpy as np
import math
import random
import copy
#######################setup

#actions are in this order: Up, Down, Left, Right

rewards=[0,-1] #possible rewards at each state

p=[] #p[present state][actions][next state][rewards]
for i in range(16):#s(0-15: 0, 15 are terminal states)
    hold=[]
    for j in range(4):#a(nswe)
        l=[]
        for k in range(16):#s'(0-15)
            m=[0 for o in range(2)] #r(0,-1)
            l.append(m)            
        hold.append(l)
    p.append(hold)
    
        
for i in range(len(p)):
    for j in range(len(p[i])):
        if(i==0):
            p[i][j][i][0]=1
        elif(i==15):
            p[i][j][i][0]=1
        else:    
            if(j==0):
                if(i-4<0):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i-4][1]=1
            elif(j==1):
                if(i+4>15):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i+4][1]=1
            elif(j==2):
                if(i==0 or i==4 or i==8 or i==12):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i-1][1]=1   
            elif(j==3):
                if(i==3 or i==7 or i==11 or i==15):
                    p[i][j][i][1]=1
                else:
                    p[i][j][i+1][1]=1
                
        
        
#just verification of the MDP setup
for i in range(len(p)):
    for j in range(len(p[i])):
        su=0;count=1;
        for k in range(len(p[i][j])):
            for o in range(len(p[i][j][k])):
#                 print(k,o,p[i][j][k][o],count);count+=1
                su+=p[i][j][k][o]
#         if(su!=1):
#             print(i,j,su)
        assert(su==1)
        

#policy initialization - equiprobable
policy=[] #policy[present state][actions] 
for i in range(16):#present states
    given_State=[]
    for i in range(4):#actions
        given_State.append(0.25)
    policy.append(given_State)

gamma=1 #discounting
    
###########################iterations setup
        
epsilon=0.00001 #convergence tolerance
    
state_value=[random.random() for i in range(16)]#initialization
state_value[0]=0;state_value[15]=0#terminal states
    
###########value iteration

def val_loop(p,state_value,gamma,epsilon,rewards,policy,iteration):
    while(True):
        
        delta=-math.inf #delta for convergence checking
        for i in range(16):#present state
            old=state_value[i]

            temp=-math.inf
            for o in range(4):#actions
                su=0;
                for j in range(16):#next states
                    for k in range(2):#rewards
                        su+=(rewards[k]+(gamma*state_value[j]))*p[i][o][j][k]
                temp=max(temp,su)
            state_value[i]=temp
            delta=max(delta,abs(old-state_value[i]))
        
        print("iteration",iteration,"value iter")#for output logging
        print(state_value)
        iteration+=1
        
        if(delta<epsilon):
            break
    
    #optimal policy generation based on the optimal state_values
    for i in range(16):#present states
        ac_values=[0 for i in range(4)] #action values calculated
        for j in range(4):#actions
            for k in range(16):#next state
                for o in range(2):#rewards
                    ac_values[j]+=p[i][j][k][o]*(rewards[o]+(gamma*state_value[k]))

        #only all optimal actions assigned equal non-zero probabilities
        act=np.argmax(ac_values)
        for j in range(4):#actions
            policy[i][j]=0
            if(ac_values[act]==ac_values[j]):
                policy[i][j]=1
        
        su=sum(policy[i])
        for j in range(4):#actions
            policy[i][j]/=su
    
    return [policy,state_value]



###########policy iteration
        
#policy evaluation
def pol_eval(p,policy,state_value,gamma,epsilon,rewards,iteration): #make these globals, function parameters
    prev=copy.deepcopy(state_value) #bug of 4.4 fixed
    
    while(True):
        
        delta=-math.inf #delta for convergence checking
        for i in range(16):#present state
            old=state_value[i]

            su=0
            for o in range(4):#actions
                for j in range(16):#next states
                    for k in range(2):#rewards
                        su+=(rewards[k]+(gamma*state_value[j]))*p[i][o][j][k]*policy[i][o]
            
            state_value[i]=su #state value of state i update in place
            
            delta=max(delta,abs(old-state_value[i]))
        
        if(delta<epsilon):
            break
    
    print("iteration",iteration,"policy eval")#for output logging
    print(state_value)
    
    if(prev==state_value): #bug of 4.4 fixed
        return [policy, state_value]
    else:  
        return pol_improve(p,policy,state_value,gamma,epsilon,rewards,iteration)
        
        
#policy improvement
def pol_improve(p,policy,state_value,gamma,epsilon,rewards,iteration):
    
    old_pol=copy.deepcopy(policy)
    for i in range(16):#present state
        ac_values=[0 for i in range(4)] #action values calculated
        for j in range(4):#actions
            for k in range(16):#next state
                for o in range(2):#rewards
                    ac_values[j]+=p[i][j][k][o]*(rewards[o]+(gamma*state_value[k]))
        
        #only all optimal actions assigned equal non-zero probabilities
        act=np.argmax(ac_values)
        for j in range(4):#actions
            policy[i][j]=0
            if(ac_values[act]==ac_values[j]):
                policy[i][j]=1

        su=sum(policy[i])
        for j in range(4):#actions
            policy[i][j]/=su
    
    print("iteration",iteration,"policy improve")#for output logging
    print(policy)
    iteration+=1
    
    if(policy!=old_pol):
        return pol_eval(p,policy,state_value,gamma,epsilon,rewards,iteration)
    else:
        return [policy, state_value]
        


# In[7]:


#Q6) value iterations [before running this block. run cell third from the top]
[new_policy, new_state_value]=val_loop(p,state_value,gamma,epsilon,rewards,policy,0)

print();print("final policy");print(new_policy);print();print("final state values");print(new_state_value)


# In[9]:


#Q6) policy iterations [before running this block. run cell third from the top]
[new_policy, new_state_value]=pol_eval(p,policy,state_value,gamma,epsilon,rewards,0);
   
print();print("final policy");print(new_policy);print();print("final state values");print(new_state_value)

