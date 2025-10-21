I performed the measurements using 5 warm-up runs followed by 25 recorded runs for each data point. The results are summarized as mean and standard deviation in the form of a boxplot.

For **intra-node reductions**, data is exchanged via **NVLink interconnects** between GPUs, which offer a theoretical bidirectional bandwidth of up to **300 GB/s**. This is significantly faster than the node-to-node network, explaining the much higher performance observed in this configuration.

When process groups are defined so that each rank resides on a different node, **NVLink** can no longer be utilized. In this setup, the benchmark effectively measures the **node-to-node network performance**—in this case, using **Cray’s Slingshot-11 interconnect**—which is considerably slower than NVLink. It would be interesting to compare these results with systems using other topologies, such as **FZ Jülich’s Dragonfly+**, where smaller jobs are placed within groups that provide roughly twice the bandwidth of Slingshot-11.

A noticeable **bandwidth jump** occurs between message sizes of \(2^{18}\) and \(2^{19}\) elements. My hypothesis is that this reflects a change in the underlying **MPI communication protocol**, switching from a latency-optimized approach for smaller messages to a bandwidth-optimized one for larger messages (e.g., transitioning between buffered and unbuffered modes). This explanation, however, remains speculative without concrete evidence.

Finally, the **global reduction** operation leverages both intra- and inter-node communication. As expected, its performance falls between that of the pure intra-node and pure inter-node cases.
