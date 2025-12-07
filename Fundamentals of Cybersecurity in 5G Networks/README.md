# Lab for Unit 3.1 - Fundamentals of Cybersecurity in 5G Networks

This lab is part of Module 3 (Cybersecurity challenges and solutions in 5G networks) of the "Digital Skills for Beyond 5G Technologies" course from the 5G-DiGITS project. It provides a hands-on experience with the concepts discussed in the course materials, focusing on simulating a 5G network environment with MEC (Multi-access Edge Computing) and IoT components.

## Case Study

This lab is associated with a case study presented in the course. The `doc` folder contains the complete course as a Word document and three presentations that explain the case study. This lab allows students to download and test the concepts from the case study in a simulated environment.

## Lab Scenario

The lab environment simulates a simplified 5G network with the following components:
*   A 5G core network (Open5GS).
*   An MQTT broker for IoT communications.
*   A MEC application stub that exposes a control API.
*   An attacker container with scripts to launch simulated attacks.
*   A monitoring stack with Prometheus and Grafana.
*   An Intrusion Detection System (IDS) with Suricata.

The goal of this lab is to understand how to identify and detect common security threats in a 5G environment, such as Denial of Service (DoS) attacks and unauthorized API access.

## Getting Started

To run this lab, you will need to have Docker and Docker Compose installed on your system.

1.  Clone this repository to your local machine.
2.  Navigate to the `lab/compose` directory.
3.  Run the command `docker-compose up --build -d` to build the container images and start the lab environment in detached mode. The `--build` flag is important to ensure that the latest changes in the Dockerfiles are applied.
4.  To stop the lab, run `docker-compose down`.

## Services

The following services are part of this lab and can be accessed via the specified ports on your localhost:

*   **Open5GS WebUI:** `http://localhost:3000`
*   **MEC Stub API:** `http://localhost:8080`
*   **Prometheus:** `http://localhost:9090`
*   **Grafana:** `http://localhost:3001` (user: admin, password: admin)
*   **Mosquitto MQTT:** `localhost:1883`

## Project Structure

The `lab` folder contains the following directories and files:

*   **`attacker-scripts/`**: This directory contains scripts and the Dockerfile for the attacker container.
    *   `api_bypass.sh`: A script that sends multiple unauthenticated POST requests to the MEC application's `/control` endpoint, simulating an attempt to bypass API authentication or authorization.
    *   `dos_test.sh`: A script that uses `hping3` to launch a low-rate SYN flood attack against the MEC application, simulating a Denial of Service (DoS) attack.
    *   `Dockerfile`: Builds the attacker container with necessary tools like `curl` and `hping3`.
*   **`compose/`**: This directory contains the Docker Compose configuration for the lab.
    *   `docker-compose.yml`: The main Docker Compose file that defines all the services, their builds, ports, volumes, and networks.
    *   `prometheus.yml`: Configuration file for Prometheus, defining the scrape targets (in this case, the MEC stub's `/metrics` endpoint).
*   **`mec-stub/`**: This directory contains a stub for a Multi-access Edge Computing (MEC) application.
    *   `app.py`: A simple Flask web application that simulates a MEC service with `/health`, `/control`, and `/metrics` endpoints. It also exposes Prometheus metrics.
    *   `Dockerfile`: Builds the MEC stub container and runs the Flask application.
*   **`sensor-sim/`**: This directory is intended for a sensor simulator, but it is currently empty.
*   **`suricata-rules/`**: This directory contains custom rules for the Suricata Intrusion Detection System (IDS).
    *   `suricata.rules`: A file containing custom Suricata rules to detect the simulated attacks. The rules are designed to detect DoS floods and suspicious access to the MEC API and MQTT broker.

## Course Information

*   **Course:** Digital Skills for Beyond 5G Technologies
*   **Project:** 5G-DiGITS
*   **Module:** 3 - Cybersecurity challenges and solutions in 5G networks
*   **Unit:** 3.1 - Fundamentals of Cybersecurity in 5G Networks
