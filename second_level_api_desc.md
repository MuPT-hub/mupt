# Second level API

This intermediate API is meant to (1) give users more control over generating polymers while not forcing them to dig into the internals of MuPT and (2) to offer convinent and easy-to-use mathematical models for the structure of polymers, including polydispersity, tacticity, and copolymer sequence.

The API is solely contained within the `mupt.polymers` module, and is built around a central `TopologyGenerator` class, which constructs descriptive graphs of polymer systems. There are two  types of `TopologyGenerators`: simple and composite. Simple generators build a descriptive graph of a system where nodes represent repeat units and edges represent bonds. Composite generators take in simple generators and build descriptive graphs at a coarse resolution, where each node represents a component (a simple generator) and edge specify the connections between the components.

