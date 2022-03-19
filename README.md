# AI-based-AudioVis
Audio visualization is something humans generate/enjoy in many forms. Some examples: 
- Lighting at a concert 
- Creating levels in any rhythm based game (OSU, Beatsaber, etc.) 
- Syncing LED lights to music 

This works on the following 3 components:  
X: Some music is observed (heard)  
P: Human brain creates a compact internal representation of observed music  
Y: Said representation is converted to an artistic representation  
X -> P -> Y  
X and Y can be done in various ways. P, the most difficult component, can be emulated by using a representation learning model. Said model would need (X,Y) examples to be trained. This repo explores various ways to implement X (audio), P (model), and Y (visualization).

# Requirements  
For the VAE layers, I use a cuda kernel nvidia made for a signal processing function for [stylegan2](https://github.com/NVlabs/stylegan2-ada-pytorch). Download a zip of relevant code here: [link](https://drive.google.com/file/d/1QWH3_jJV65cN3ebc0yZvXJ3t_ROm3ZG3/view?usp=sharing)
