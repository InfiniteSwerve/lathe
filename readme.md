Lathe will turn a neural network into a much sparser subgraph that tries to effectively capture the computation flow in as few nodes as possible. It has the added advantage of using not only the priviliged basis as nodes, but also other directions that one might get from other sources like principle components and sparse auto encoders 
## The Algorithm:
```python
H = G
H = H.reverse_topological_sort()
threshold = tau
for node in H:
    for parent in node.parents:
        H_prime = H.remove_edge(parent, node)
        if KLDiverge(G, H_prime) - KLDiverge(G, H) < threshold:
            H = H_prime

return H
```

## NOTES:

inspired by https://arxiv.org/pdf/2304.14997.pdf
What structure are we going to use to track the graph?
- Probably whatever graphvis uses. It's probably pretty optimized. 
- How will we deal with temporarily removing an edge?
We need some correspondence between the graph and the actual matrices. 
During inference, we can just use a mask and add elemnts to the mask when we remove edges from the graph
What are the effects of removing any particular edges in any order?
If we remove things from earlier on in the computation it'll have a nonlinear effect, so we remove things at the end of the computation first because they should have linear effects at that point. 
    - I guess the question is something like, is ablation path independent if you do it starting at the end and working backwards?
    - I don't think it is, because if a relu bias starts at -2, and there are 3 neurons connected each of weight 1, then you only need 2 weights connected for the circuit to work. So you'll always be able to disconnect 1 neuron and the order that you decide to search chooses which one you'll disconnect. Does this hold true across the whole dataset given KL divergence? Does that change anything?
    - There should be some other way to calculate importance when there are multiple options..

    
We should also have some way to compare several graphs, either by union, intersection, or overlay.



So the thing I'm most uncertain about is how to actually deal with the SAE or other directions:
1. I could interpret them as more graph edges, and allow all features to be pruned. 
2. Is there a better way to start out or use gradients?
    - I think that's more complicated and I don't know how yet. Let's go with option 1. 

Given a network, there are standard ways to interpret it as a graph. But if we're in a rotated basis due to the SEA, how can we interpret that network as a graph? Is there a way to preserve the rotation? Or do we just need to stay in the standard basis?
    - Maybe that's an actual way to use the PCA that I did way back when. 

What if different subnetworks are responsible for different parts of the distribution of work
like what if tracking a token that appears in the first half of the game is handled by one subnetwork, and tracking if a token appears in the last half of a game appears in a different subnetwork. How would we distringuish these two scenarios? How would we know to carve along that?

What about finding out which subsets of the dataset are most strongly handled by some circuit, or which circuits are most strongly handled by some subset of the dataset. 

The problem of all this stuff is that it still requires so much looking and squinting at the damn thing if we don't find the right abstractions the first time. That's a really big problem here.

Given a subgraph, we can try to more directly isolate the behavior by further probing and ablation. Maybe that's an easier way to directly spot the algorithm or reverse it via causal intervention?

