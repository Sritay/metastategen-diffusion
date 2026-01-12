# Understanding Geometry Fixes

## 1. Effect on Refinement Loop
**Will this break the refinement loop?**
No.
**Will it remove the need for refinement?**
No.

**Refinement** (using the Force Surrogate) does two things:
1.  **Fixes local geometry**: Bonds, Angles (e.g.making length 0.15nm instead of 0.16nm).
2.  **Fixes Physics (Energy)**: Resolves steric clashes (atoms overlapping), optimizes hydrogen bonds, and moves the structure to a low-energy basin.

The **Bond Projection** only does part of (1). It forces the *sticks* (bonds) to be the right length. It does *not* know about energy or clashes. So you absolutely still need the Refinement Loop to make the molecule physically stable (minimize energy). Projections just make the job easier for the refiner by giving it a non-broken starting point.

## 2. What is "Constraint Projection" during Sampling?

### The Process (Simple English)
Imagine you are drawing a stick figure (the molecule) on paper.
- **Diffusion Step**: The AI guesses where the joints (atoms) should be. It draws them, but the arm is way too long (0.3nm instead of 0.15nm).
- **Projection Step**: You immediately take a ruler, measure the arm, and say "No, this must be 0.15nm." You erase the hand and redraw it closer to the shoulder so the length is perfect.
- **Repeat**: The AI sees the corrected drawing and makes its next guess based on that.

By doing this "check and fix" 1000 times (once per diffusion step), the final image is guaranteed to have arms of the correct length.

### Why does this help Training? (The Feedback Loop)
You are right that the "Oracle" just retrieves a nearby valid structure. But consider the **Active Learning Loop**:

1.  **Generation**: The model generates 1000 candidates.
    - *Without Projection*: Candidates are broken garbage (bonds stretched).
    - *With Projection*: Candidates look like valid molecules (correct bonds).
2.  **Selection**: We pick the most "uncertain" ones to label.
    - *Without Projection*: The model is uncertain about garbage. We ask the Oracle "What is this broken thing?". The Oracle gives us a valid structure $Y$. The model learns "Broken thing $X$ $\approx$ Valid thing $Y$".
    - *With Projection*: The model produces a decent structure $X$. The Oracle gives us valid structure $Y$. The model learns "Decent thing $X$ is actually $Y$".
    
    The key is that by forcing the model to explore the space of **valid-looking molecules** (correct bond lengths), we stop wasting time asking the Oracle about impossible geometries. The "Data Manifold" (the cloud of all valid molecules) is defined by these constraints. By projecting onto it, we force the model to live in reality.

### Summary
- **Inference Only**: This fix happens when the AI is *dreaming* up new structures (Sampling).
- **Training Benefit**: It creates better candidates for the next round of learning.
- **Refinement**: Still essential for physical accuracy (Energy).
