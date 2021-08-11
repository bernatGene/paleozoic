## Log

* Should start testing new code.

* There's a weird bug in food consumption. Some positions don't 
    allow for the agent to get the food. 
* ~~**Simultaneous actions or by turns???**~~
    * Simultaneous:
        * Agent map (no walls), sum overlaps
        * Init perception (using agent map + food map)
        * For all agents:
            * Take the actions and record the decisions
              (but don't move yet)
        * For all agents (In rand order):
            * Move them all on agent map
            * Keep an intersection_dict for each intersection
            encountered (bool map of occupied cells)
        * Deliver rewards and penalties
      

## Desired features:

* Allow for saving models.

* Allow for counter-clockwise rotation, otherwise some agents
can get themselves in a stalemate against the wall. 
  
* Add predation:
    ~~* Define interaction rules~~
        * ~~When an agent is predated on, the reward
          (negative) should be subtracted from the last
          move, since it is what got the agent into
          the situation~~.
        * ~~Agents can intersect, interaction rules
        define what happens when they do~~
        ~~* Be careful with how to interact with corpses.~~
    ~~* Add ability for agents to see each other~~
      ~~* Add agent intersection function
        * Intersection should be done with a map copy
      without walls and the bodies in it~~
          
    * Add dummy "herbivore" agents
* Make map bigger 
    * Better tiling system
    * Safe spawn zones
        * Define a function to generate a list of
      safe zones and assign them according to the agents
          bounding box. 
    * Grid-like agent positions to optimize 
    interaction determination
      
    * Add the possibility to transform map into png.
    * For terminal viewer, maybe allow navigating ?
        * Maybe in streamlit ?
        * Add a log of what happened on each step, for debugging. 
    
* Add dynamic perception
    * Add dynamic models (hard)
        * Models should be "mergeable"

* Add evolution
    * keep a genealogical record
  
* Design training pipeline where for each x epochs,
the latest day is shown and played back on the terminal
  * Maybe with Streamlit ? 

* Docstrings + Documentation ?

## Ideas

* Criteria to stop and evolve:
    * Agent has survived All days x times. That is the
    agent from which the next generation is created.

### On mergeable / dynamic agents

The key "cool" thing behind the whole project was to 
allow the agents to grow dynamically by genetic /
evolutionary methods. The main challenge, however, is:
how can you get a new model architecture while still
leveraging the training achieved on the parent 
architecture from which it evolved. 

I don't even know if this is possible, but at the very
least, many architectures won't be possible. 

A simple way to achieve dynamic agents with arbitrary
input size would be to have a model for each perceptive
cell, then define a function to sample from the 
decisions of each model. 

The next best thing I can think of is to have a 
"Feature vector" that is common to all agents
together with the architecture of the actor and the
critic, and this does not change.
Then, for each perception cell, a feature extractor is 
built, each with a final output shape equal to the
common feature vector. 

Then, have a learnable function
that decides how to merge the feature vectors. I'm
not clear on what the inputs to the function should
be, or how it should be trained, but at least there
should be some learnable parameters that "modulate"
the weight of each feature extractor.

To be able to "preserve" the learned weights so far,
When a new perception cell is built, and with it
its feature extractor, a new parameter appears as well
for the merging function. All other parameters are 
updated, so they share 95% of the weight with the
same proportion as they did, while the new parameter
gets 5% of the weight. 

The merging function could be as easy as a weighted 
addition /  avergage of the feature vectors, but my
intuition is that this would necessarily destroy info
since each is bringing different info. 







