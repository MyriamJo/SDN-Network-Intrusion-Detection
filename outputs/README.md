# SDN Network Intrusion Detection Using Statistical Modeling and Naive Bayes

A machine learning-based intrusion detection system for Software-Defined Networks (SDNs) using switch and port-level telemetry. The project analyzes network behavior patterns from SDN monitoring data, learns statistical traffic profiles, and classifies network activity into normal traffic or one of five attack categories.

## Overview

Software-Defined Networks provide detailed telemetry from switches and ports, including packet rates, byte counters, flow statistics, packet drops, and load measurements. These metrics change significantly during cyberattacks, creating detectable behavioral patterns even when malicious traffic attempts to appear similar to legitimate communication.

This project models SDN telemetry features as probabilistic variables and develops a statistical intrusion detection pipeline that:

- Explores network behavior patterns across different attack scenarios.
- Learns probability distributions for SDN traffic features.
- Identifies the most informative telemetry features for attack detection.
- Classifies unseen network traffic using a probabilistic Naive Bayes classifier.

The system detects six traffic classes:

- Normal traffic
- TCP-SYN attack
- PortScan attack
- Overflow attack
- Diversion attack
- Blackhole attack

---

# Dataset

The project uses the **UNR-IDD (University of Nevada, Reno Intrusion Detection Dataset)**(https://www.kaggle.com/datasets/tapadhirdas/unridd-intrusion-detection-dataset), which contains approximately 37,400 labeled SDN telemetry samples.

Each sample includes 34 features describing switch and port behavior, including:

- Packet and byte counters
- Packet and byte rate variations
- Flow table statistics
- Packet drops and errors
- Network load measurements
- Traffic labels

The target classification variable is:
