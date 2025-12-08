# COMS-E6998 Final Project

## Project Roadmap

| Date | Milestone | Status |
|------|-----------|--------|
| **October 19** | GitHub set up with example files | ✅ Complete |
| **October 26** | V1 flow working | ✅ Complete |
| **November 10** | Initialize last model for training | 🔄 In Progress |
| **November 18** | Set up model on live traffic (final deadline) | ⏳ Upcoming |
| **December 15** | Project due date (earliest possible) | ⏳ Upcoming |


## Next Steps

- Finish report rough draft
- Start slides for presentation


## Experimental Outline

1. Train a strong teacher that we can use for the remainder of the project
2. Train a simple student with no optimizations - baseline KD
3. Try an experimental KD approach such as FitNets or Attention Transfer
4. Rerain students using quantiztion aware training
    - Could also contrast with post-training quantization
5. Compare and contrast loss compared to teacher, latency, cost, ROI, etc.
    - Results would be a matrix between KD and quantization techniques
