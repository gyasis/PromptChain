# Project Requirements Document: Zenfl Job Alert Interceptor & Processor

**Project Name:** Zenfl Job Alert Interceptor

**Version:** 1.0

**Author:** [Your Name/Team]

**Date:** [Current Date]

## 1. Introduction

### 1.1 Project Goal
The primary objective of this project is to create a self-hosted, custom Telegram client (a "user bot") that acts as a real-time message interceptor and processor for job alerts sent by the Zenfl bot. This solution will eliminate the need for third-party subscription services and provide a high degree of control over how job alerts are managed and presented.

### 1.2 Problem Statement
Using a pre-built service like Zenfl is convenient, but it limits a user's ability to automate and customize the data flow. Manually forwarding messages is inefficient, and paid forwarding services incur ongoing subscription costs. A custom-built user bot, powered by the low-level Telegram API, offers a robust and free solution for a user with coding skills.

## 2. High-Level Architecture

The project will consist of a single application that will be hosted on a remote server. The application will be built using a Python library that provides an interface to the Telegram API (such as TDLib, Telethon, or Pyrogram).

### Components:

- **Telegram API Client:** The core of the application, responsible for authenticating as a user and listening for real-time messages.

- **Message Interceptor:** A component that filters all incoming messages, specifically looking for those originating from the Zenfl bot.

- **Data Parser:** A module to extract key data (e.g., job title, URL, budget, keywords) from the plain text of Zenfl's messages.

- **Processor & Action Manager:** The logic that determines what to do with the parsed data (e.g., send to a new channel, log to a file).

## 3. Technical Requirements

### 3.1 Core Technology Stack
- **Programming Language:** Python (recommended for its extensive libraries and community support).

- **Telegram API Library:** A low-level library that can emulate a Telegram user account. Options include:
  - **TDLib (Telegram Database Library):** Telegram's official client library. Requires a more complex build process but is officially maintained.
  - **Telethon:** A popular, well-documented, and high-level Python library that uses the Telegram API directly. It's often easier to get started with than TDLib's native bindings.
  - **Pyrogram:** Another powerful, modern, and asynchronous Python library for the Telegram API.

- **Hosting:** A reliable environment for 24/7 execution, such as a Virtual Private Server (VPS), a cloud service (e.g., Heroku, DigitalOcean), or a small computer like a Raspberry Pi.

### 3.2 Authentication & API Credentials
- **Telegram API api_id and api_hash:** These credentials are required to use the low-level Telegram API. They must be obtained by registering a new application on the Telegram Developer website (https://my.telegram.org/apps).

- **User Account:** The script will require a phone number to log in as a user. It must handle the one-time authentication process, including receiving and inputting the authentication code.

- **Session File:** To avoid re-authentication, the script must be able to securely store the session data in a file (.session) and reuse it on subsequent runs.

### 3.3 Functional Logic
- **Message Reception:** The application must start a client session and continuously listen for UpdateNewMessage events from the Telegram servers.

- **Filtering:** Upon receiving a new message, the script will check the sender's ID (from_user_id) to see if it matches the Zenfl bot's ID.

- **Parsing:** Using string manipulation, regular expressions, or other parsing techniques, the script will extract structured data from the text of a Zenfl job alert.

- **Conditional Processing:** The script must support customizable logic to:
  - Filter out unwanted jobs based on keywords (e.g., "marketing," "SEO"), minimum budget, or job type.
  - Format the message into a cleaner, more readable output.
  - Send the final, filtered message to a specific chat, channel, or group.

- **Persistence:** To avoid sending duplicate notifications, the script must store a history of recently processed job IDs (e.g., in a simple text file or a small database).

## 4. Implementation Plan

### Setup and Installation:
1. Install Python and a suitable Telegram API library (e.g., `pip install telethon`).
2. Register a new app on my.telegram.org to get your api_id and api_hash.

### Initial Scripting:
1. Write a basic script to authenticate a user account using the obtained API credentials.
2. Implement the session management to persist the login.
3. Verify that the script can connect and send a test message to a user.

### Message Interception:
1. Write a message handler to listen for incoming messages in real-time.
2. Add a conditional check to specifically target messages from the Zenfl bot's user ID.

### Parsing & Data Extraction:
1. Analyze the structure of Zenfl's job alerts.
2. Develop a parsing function to extract the job title, link, and other details.

### Automation Logic:
1. Implement the filtering logic based on project needs (e.g., keyword match, budget check).
2. Add a function to send the final, processed message to a desired Telegram destination.
3. Implement a simple persistence layer to store and check for duplicate jobs.

### Deployment:
1. Set up a VPS or other cloud service.
2. Deploy the Python script to the server.
3. Use a process manager like systemd or Supervisor to ensure the script runs continuously and automatically restarts if it crashes.

## 5. Final Output

The final deliverable will be a running, self-hosted Python script that automatically monitors your Telegram chat, intercepts Zenfl job alerts, and sends a filtered and formatted notification to a destination of your choosing without any manual intervention or recurring subscription fees.
