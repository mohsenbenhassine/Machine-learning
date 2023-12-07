import numpy as np

def Q_learn_route(FromLoc,Final_Loc):
     
    # Put Max reward to the ending state to avoid infinite loop
    ending_state = Final_Loc
    rewards[ending_state,ending_state] = 999

    # Q-Learning state selection process
    for i in range(episodes):
        # Select a state randomly
        current_state = np.random.randint(0,9)  
        possible_actions = []
        
        # insert possible actions for the current state, where the rewards > 0
        for j in range(9):
            if rewards[current_state,j] > 0:
                possible_actions.append(j)
        
        # Select an action randomly from possible actions (explore approach)  
        next_state = np.random.choice(possible_actions)
        
        # Compute the temporal difference
        TD = rewards[current_state,next_state] + gamma * Q[next_state,np.argmax(Q[next_state,])] - Q[current_state,next_state]
        
        # Update the current Q-Value using the Bellman equation
        Q[current_state,next_state] += alpha * TD

    # Initialize the best route with the starting room
    Best_route = rooms[FromLoc]
    next_location = FromLoc
    
    
    while(next_location != Final_Loc):
        # Fetch the actual state
        starting_state =  FromLoc 
        # Select the highest Q-value of all the next possible states
        next_state = np.argmax(Q[starting_state,])
        
        # Replace actual state by best next state and update best route
        next_location =  next_state
        Best_route+=" "+rooms[next_location]
        
        # Update the actual state 
        FromLoc = next_location
    
    return Best_route

rooms={0:"R1",1:"R2",2:"R3",3:"R4",4:"R5",5:"R6",6:"R7",7:"R8",8:"R9" }
states = [0,1,2,3,4,5,6,7,8]
rewards = np.array([[0,1,0,1,0,0,0,0,0],
              [1,0,1,0,1,0,0,0,0],
              [0,1,0,0,0,0,0,0,0],
              [1,0,0,0,1,0,0,0,0],
              [0,1,0,1,0,1,0,1,0],
              [0,0,0,0,1,0,0,0,1],
              [0,0,0,0,0,0,0,1,0],
              [0,0,0,0,1,0,1,0,1],
              [0,0,0,0,0,1,0,1,0]])
# Discount factor 
gamma = 0.6
# Learning rate 
alpha = 1 
# initialize Q matrix by zeros
Q = np.array(np.zeros([9,9]))

# Get Start and final room entries
i=int(input("Best room From room #:"))
j=int(input("          To room #:"))
i-=1
j-=1
episodes=3000
print("Best route:",Q_learn_route(i, j))