# Athena LightRAG MCP Server - Usage Examples

## Overview

This guide provides practical examples of using the Athena LightRAG MCP Server for various healthcare database analysis scenarios. Each example demonstrates different query modes, reasoning strategies, and use cases.

## Basic Query Examples

### Example 1: Table Structure Inquiry (Local Mode)

**Scenario:** You need specific technical details about database tables.

```json
{
  "tool": "query_athena",
  "parameters": {
    "query": "What are the columns and data types in the patient_appointments table?",
    "mode": "local",
    "top_k": 15
  }
}
```

**Expected Response:**
```
The patient_appointments table contains the following columns:
- appointment_id (INTEGER, PRIMARY KEY): Unique identifier for each appointment
- patient_id (INTEGER, FOREIGN KEY): Links to patients table
- provider_id (INTEGER, FOREIGN KEY): Links to healthcare_providers table
- appointment_datetime (TIMESTAMP): Scheduled date and time
- appointment_type_id (INTEGER, FOREIGN KEY): Links to appointment_types table
- status (VARCHAR(20)): Current status (scheduled, completed, cancelled, no_show)
- duration_minutes (INTEGER): Expected appointment duration
- notes (TEXT): Additional appointment notes
- created_at (TIMESTAMP): Record creation timestamp
- updated_at (TIMESTAMP): Last modification timestamp
```

**When to use Local Mode:**
- Specific table schemas
- Field definitions and relationships
- Technical implementation details

### Example 2: System Overview (Global Mode)

**Scenario:** You need a high-level understanding of the database structure.

```json
{
  "tool": "query_athena",
  "parameters": {
    "query": "What are the main categories of tables in the medical database?",
    "mode": "global",
    "top_k": 30
  }
}
```

**Expected Response:**
```
The Athena medical database is organized into several main categories:

1. **Patient Management Tables**
   - Patient demographics, registration, and contact information
   - Insurance and eligibility data

2. **Clinical Operations Tables**
   - Appointment scheduling and management
   - Provider and staff information
   - Clinical documentation and notes

3. **Medical Procedures Tables**
   - Surgical procedures and anesthesia records
   - Diagnostic tests and imaging
   - Treatment plans and protocols

4. **Financial and Billing Tables**
   - Billing transactions and payment processing
   - Insurance claims and reimbursements
   - Revenue cycle management

5. **Administrative Tables**
   - System configuration and lookup values
   - Audit logs and compliance tracking
```

**When to use Global Mode:**
- System architecture overview
- Understanding major functional areas
- Initial database exploration

### Example 3: Balanced Analysis (Hybrid Mode)

**Scenario:** You need both specific details and broader context.

```json
{
  "tool": "query_athena",
  "parameters": {
    "query": "How does patient scheduling integrate with provider availability and billing systems?",
    "mode": "hybrid",
    "top_k": 40
  }
}
```

**Expected Response:**
```
Patient scheduling integration involves multiple interconnected systems:

**Core Scheduling Components:**
- patient_appointments table manages scheduled visits
- scheduling_slots table tracks provider availability windows
- appointment_types table defines procedure categories and durations

**Provider Integration:**
- healthcare_providers table contains staff schedules and specialties
- provider_availability table manages time slot allocation
- Cross-references ensure appropriate provider-patient matching

**Billing System Connection:**
- appointment_billing_events table links appointments to financial records
- procedure_codes table maps appointments to billable services
- billing_transactions table processes payments and insurance claims
- Revenue recognition occurs upon appointment completion or service delivery

**Data Flow:**
1. Patient requests appointment → system checks provider availability
2. Appointment scheduled → billing codes assigned based on procedure type
3. Appointment completed → billing event triggered
4. Insurance processed → payment reconciled

This integrated approach ensures seamless coordination between clinical operations and financial management.
```

**When to use Hybrid Mode:**
- Most general queries
- Understanding system interactions
- Balancing detail with context

### Example 4: Simple Search (Naive Mode)

**Scenario:** You need basic keyword-based retrieval without complex reasoning.

```json
{
  "tool": "query_athena",
  "parameters": {
    "query": "anesthesia billing codes",
    "mode": "naive",
    "top_k": 10
  }
}
```

**Expected Response:**
```
Found references to anesthesia billing codes in the following areas:

- anesthesia_billing_codes table contains specific procedure codes for anesthesia services
- anesthesia_procedures table references billing code assignments
- billing_transactions table includes anesthesia service charges
- procedure_codes table contains standardized anesthesia billing identifiers
- Claims processing workflows include anesthesia-specific billing rules
```

**When to use Naive Mode:**
- Simple keyword searches
- Quick fact-finding
- When other modes are too complex

## Multi-hop Reasoning Examples

### Example 5: Comprehensive Workflow Analysis

**Scenario:** Understanding complex end-to-end processes requiring multiple analysis steps.

```json
{
  "tool": "query_athena_reasoning",
  "parameters": {
    "query": "Analyze the complete patient journey from initial registration through surgical procedure and final billing, identifying all key integration points and potential bottlenecks",
    "context_strategy": "comprehensive",
    "max_reasoning_steps": 6
  }
}
```

**Expected Reasoning Process:**

**Step 1: Patient Registration Analysis**
```
Analyzing patient registration workflow:
- patient_demographics table captures initial patient information
- patient_insurance table manages coverage verification
- eligibility_verification table tracks insurance status
Key integration: Registration system connects to insurance verification APIs
```

**Step 2: Scheduling System Analysis**
```
Examining appointment scheduling process:
- patient_appointments table manages scheduling
- provider_availability integrates with staff schedules
- surgical_scheduling table handles OR booking
Integration point: Scheduling system coordinates with provider management
```

**Step 3: Pre-procedure Workflow**
```
Analyzing pre-surgical preparation:
- pre_procedure_checklist table tracks patient preparation
- anesthesia_assessment table captures pre-op evaluation
- surgical_consent table manages documentation
Critical integration: Clinical systems coordinate with surgical scheduling
```

**Step 4: Surgical Procedure Execution**
```
Examining intraoperative workflow:
- surgical_procedures table tracks operation details
- anesthesia_records table captures anesthesia administration
- OR_utilization table manages resource allocation
Key integration: Real-time data flows between clinical systems
```

**Step 5: Post-procedure and Discharge**
```
Analyzing post-operative process:
- recovery_room_records table tracks patient status
- discharge_planning table manages patient release
- follow_up_appointments table schedules continuing care
Integration point: Clinical outcomes feed into continuing care planning
```

**Step 6: Billing and Revenue Cycle**
```
Examining financial workflow completion:
- procedure_charges table captures all billable services
- insurance_claims table manages reimbursement requests
- payment_processing table handles collections
Final integration: Clinical documentation drives billing accuracy
```

**Synthesized Analysis:**
```
COMPLETE PATIENT JOURNEY ANALYSIS

**End-to-End Workflow:**
Registration → Scheduling → Pre-procedure → Surgery → Recovery → Billing

**Key Integration Points:**
1. Registration-Insurance Interface: Real-time eligibility verification
2. Scheduling-Provider Coordination: Dynamic availability management
3. Clinical-Surgical Integration: Seamless information flow
4. Procedure-Billing Connection: Automated charge capture
5. Documentation-Compliance Link: Regulatory requirement tracking

**Potential Bottlenecks Identified:**
1. **Insurance Verification Delays**: Manual verification processes can slow registration
2. **Provider Scheduling Conflicts**: Limited OR availability creates scheduling constraints
3. **Clinical Documentation Gaps**: Incomplete records delay billing processes
4. **System Integration Issues**: Data silos between clinical and financial systems
5. **Discharge Planning Coordination**: Multiple departments must coordinate patient release

**Optimization Recommendations:**
- Implement automated insurance verification APIs
- Deploy real-time provider scheduling coordination
- Establish integrated clinical documentation workflows
- Create automated billing trigger mechanisms
- Develop centralized discharge planning coordination system
```

### Example 6: Incremental Problem Analysis

**Scenario:** Building understanding step-by-step through focused inquiry.

```json
{
  "tool": "query_athena_reasoning",
  "parameters": {
    "query": "Why might there be discrepancies between scheduled appointment durations and actual procedure times, and how does this affect downstream billing and scheduling?",
    "context_strategy": "incremental",
    "max_reasoning_steps": 4
  }
}
```

**Reasoning Chain:**

**Step 1: Duration Discrepancy Investigation**
```
Investigating appointment duration vs actual time differences:

Query: "What factors influence appointment duration estimates vs actual times?"

Analysis: The appointment_types table contains standard duration estimates, but actual_procedure_duration in surgical_procedures table often varies due to:
- Patient complexity factors not captured in initial scheduling
- Unexpected complications during procedures
- Provider experience and efficiency variations
- Equipment availability and setup time differences
```

**Step 2: Impact on Scheduling Systems**
```
Building on duration analysis, examining scheduling impacts:

Query: "How do duration discrepancies affect subsequent appointment scheduling?"

Analysis: When procedures run over scheduled time, it creates cascading effects:
- scheduling_slots table shows availability conflicts
- provider_schedule_adjustments table tracks real-time changes
- OR_utilization_tracking shows efficiency metrics declining
- Downstream appointments may need rescheduling, affecting patient satisfaction
```

**Step 3: Billing System Effects**
```
Extending analysis to financial implications:

Query: "How do time discrepancies impact billing accuracy and revenue?"

Analysis: Duration variations affect billing through:
- procedure_charges table may have time-based billing components
- anesthesia_billing_codes table often includes time-dependent charges
- OR_facility_fees table charges based on actual usage time
- Insurance reimbursement may be affected by procedure complexity indicators
```

**Step 4: Systemic Solution Analysis**
```
Synthesizing findings to identify systemic solutions:

Final Analysis: To address duration discrepancies and their downstream effects:

**Root Causes:**
- Inadequate initial patient assessment data
- Static scheduling assumptions vs dynamic procedure reality
- Limited real-time adjustment capabilities

**Systematic Solutions:**
1. Enhanced pre-procedure assessment protocols
2. Dynamic scheduling with buffer time allocation
3. Real-time procedure tracking and adjustment systems
4. Automated billing adjustments based on actual times
5. Predictive analytics for better duration estimation

**Implementation Priority:**
- Immediate: Buffer time scheduling adjustments
- Short-term: Real-time tracking system deployment
- Long-term: Predictive analytics integration
```

### Example 7: Focused Deep Dive Analysis

**Scenario:** Concentrated analysis of a specific system component.

```json
{
  "tool": "query_athena_reasoning",
  "parameters": {
    "query": "Perform a detailed analysis of the anesthesia workflow system, focusing on how it integrates with surgical scheduling, clinical documentation, and billing processes",
    "context_strategy": "focused",
    "max_reasoning_steps": 4
  }
}
```

**Focused Reasoning Process:**

**Step 1: Anesthesia Workflow Core Analysis**
```
Deep dive into anesthesia-specific workflow components:

Core Tables Analysis:
- anesthesia_assignments: Links anesthesiologists to surgical cases
- anesthesia_records: Captures detailed intraoperative anesthesia data
- anesthesia_medications: Tracks drug administration and dosages
- anesthesia_monitoring: Records vital signs and patient responses

Key Workflow: Pre-op assessment → Equipment setup → Induction → Maintenance → Emergence → Recovery handoff
```

**Step 2: Surgical Scheduling Integration**
```
Analyzing how anesthesia integrates with surgical scheduling:

Integration Points:
- surgical_schedule_anesthesia table coordinates provider assignments
- anesthesia_equipment_requests links to OR setup requirements
- procedure_anesthesia_requirements specifies needed anesthesia types
- anesthesia_provider_availability aligns with surgical case scheduling

Critical Dependencies:
- Anesthesiologist availability must align with surgeon schedules
- Equipment requirements must be reserved in advance
- Patient-specific anesthesia plans affect OR preparation time
```

**Step 3: Clinical Documentation Integration**
```
Examining anesthesia documentation within broader clinical workflow:

Documentation Flow:
- pre_anesthesia_evaluation feeds into surgical planning
- intraoperative_anesthesia_record integrates with surgical notes
- post_anesthesia_recovery_record coordinates with nursing documentation
- anesthesia_complications_tracking links to quality improvement systems

Compliance Requirements:
- Regulatory documentation standards require complete anesthesia records
- Quality metrics depend on accurate anesthesia outcome tracking
- Medical-legal documentation requires comprehensive anesthesia monitoring data
```

**Step 4: Billing Process Integration**
```
Deep analysis of anesthesia billing complexity:

Billing Components:
- anesthesia_base_units: Standard procedure complexity scoring
- anesthesia_time_units: Duration-based billing calculations
- anesthesia_qualifying_circumstances: Additional complexity factors
- anesthesia_physical_status_modifiers: Patient condition adjustments

Revenue Cycle Integration:
- Anesthesia charges calculated using base + time + modifying factors
- Insurance prior authorization requirements for complex cases
- Claims processing requires accurate procedure and time documentation
- Revenue recognition tied to complete anesthesia documentation

**Synthesis - Comprehensive Anesthesia System Integration:**

The anesthesia workflow represents one of the most complex integrations in the healthcare system:

**Scheduling Integration:** Anesthesia provider assignment must coordinate with surgical scheduling, considering provider credentials, patient needs, and equipment requirements.

**Clinical Integration:** Anesthesia documentation flows seamlessly through pre-operative assessment, intraoperative monitoring, and post-operative care, integrating with all clinical systems.

**Financial Integration:** Anesthesia billing involves complex calculations based on time, complexity, and patient factors, requiring accurate integration between clinical documentation and billing systems.

**Quality Integration:** Anesthesia outcomes data feeds into quality improvement initiatives, complication tracking, and regulatory reporting requirements.

This multi-dimensional integration makes anesthesia workflow optimization critical for overall healthcare system efficiency.
```

## Context-Only Queries

### Example 8: Retrieving Raw Context

**Scenario:** You need the underlying knowledge base information without LLM processing.

```json
{
  "tool": "query_athena",
  "parameters": {
    "query": "patient scheduling workflow",
    "mode": "hybrid",
    "context_only": true,
    "top_k": 20
  }
}
```

**Expected Response:**
```
[Raw Knowledge Base Context]

Entity: patient_appointments
- Primary scheduling table
- Links patients to providers and time slots
- Status tracking for appointment lifecycle

Relationship: patient_appointments -> patients
- Foreign key: patient_id
- Ensures referential integrity
- Enables patient demographic access

Entity: scheduling_slots  
- Manages provider availability
- Time-based resource allocation
- Capacity planning data

Relationship: scheduling_slots -> healthcare_providers
- Foreign key: provider_id  
- Links availability to specific providers
- Enables schedule coordination

Entity: appointment_types
- Defines procedure categories
- Duration estimates for scheduling
- Billing code associations

[Additional raw knowledge graph entries...]
```

**When to use context_only:**
- Building custom processing workflows
- Understanding knowledge base structure
- Developing specialized analysis tools

## Database Status and Help Examples

### Example 9: Database Health Check

```json
{
  "tool": "get_database_status",
  "parameters": {}
}
```

**Expected Response:**
```
Database Path: ./athena_lightrag_db
Database Exists: True
Initialized: True
Total Size: 125.3 MB
Total Files: 847
Database Files: entities.json, relationships.json, communities.json, graph_chunk_entity_relation.json
```

### Example 10: Query Mode Guidance

```json
{
  "tool": "get_query_mode_help",
  "parameters": {}
}
```

**Expected Response:**
```
🔍 LightRAG Query Modes Guide:

📍 LOCAL MODE:
- Best for: Specific entity relationships, detailed technical questions
- Use when: Asking about particular tables, fields, or specific components
- Example: "What are the columns in the patient_appointments table?"

🌍 GLOBAL MODE:
- Best for: High-level overviews, system-wide analysis, summaries  
- Use when: Asking about overall architecture, general patterns, broad topics
- Example: "What are the main categories of tables in the database?"

⚡ HYBRID MODE (Default):
- Best for: Most questions, combines local detail with global context
- Use when: Unsure which mode to use, or need both specific and general info
- Example: "How does patient scheduling integrate with billing systems?"

🎯 NAIVE MODE:
- Best for: Simple keyword searches, when other modes are too complex
- Use when: Looking for basic text matches without graph reasoning
- Example: Simple searches that don't require relationship understanding

🧠 Context Accumulation Strategies:

📈 INCREMENTAL:
- Builds context step-by-step through the reasoning chain
- Good for: Sequential analysis, following logical progressions

📊 COMPREHENSIVE:
- Gathers broad context from multiple perspectives
- Good for: Complex system analysis, understanding interconnections

🎯 FOCUSED:
- Targets specific areas with deep analysis  
- Good for: Specialized technical questions, detailed investigations

💡 Tips:
- Use basic queries for straightforward questions
- Use multi-hop reasoning for complex analysis requiring multiple steps
- Check database status if queries are failing
- Higher top_k values retrieve more context but may be slower
```

## Common Use Case Patterns

### Healthcare System Analysis

**Use Case: Understanding Patient Flow**
```json
{
  "tool": "query_athena_reasoning",
  "parameters": {
    "query": "Map the complete patient flow from registration through discharge, identifying all touchpoints where data is captured and how systems interact",
    "context_strategy": "comprehensive",
    "max_reasoning_steps": 5
  }
}
```

**Use Case: Revenue Cycle Optimization**
```json
{
  "tool": "query_athena_reasoning", 
  "parameters": {
    "query": "Analyze the revenue cycle workflow to identify potential bottlenecks, inefficiencies, and opportunities for automation between clinical documentation and billing processes",
    "context_strategy": "incremental",
    "max_reasoning_steps": 4
  }
}
```

### Technical Implementation

**Use Case: Database Schema Understanding**
```json
{
  "tool": "query_athena",
  "parameters": {
    "query": "What are the key foreign key relationships between patient, appointment, and billing tables?",
    "mode": "local",
    "top_k": 25
  }
}
```

**Use Case: Integration Planning**
```json
{
  "tool": "query_athena_reasoning",
  "parameters": {
    "query": "If we need to integrate a new electronic medical records system, what are the critical data integration points and potential challenges based on the current database structure?",
    "context_strategy": "comprehensive",
    "max_reasoning_steps": 5
  }
}
```

### Compliance and Quality

**Use Case: Audit Trail Analysis**
```json
{
  "tool": "query_athena",
  "parameters": {
    "query": "What audit and compliance tracking mechanisms are built into the database structure?",
    "mode": "global",
    "top_k": 30
  }
}
```

**Use Case: Data Quality Assessment**
```json
{
  "tool": "query_athena_reasoning",
  "parameters": {
    "query": "Analyze the data quality controls and validation mechanisms across clinical and financial workflows, identifying potential gaps in data integrity",
    "context_strategy": "focused", 
    "max_reasoning_steps": 4
  }
}
```

## Advanced Query Techniques

### Parameter Optimization

**High-Speed Queries:**
```json
{
  "query": "Quick table lookup",
  "mode": "naive",
  "top_k": 5
}
```

**Comprehensive Analysis:**
```json
{
  "query": "Complex system analysis",
  "mode": "hybrid",
  "top_k": 80,
  "context_strategy": "comprehensive",
  "max_reasoning_steps": 6
}
```

**Focused Deep Dive:**
```json
{
  "query": "Detailed component analysis",
  "mode": "local", 
  "top_k": 40,
  "context_strategy": "focused",
  "max_reasoning_steps": 3
}
```

### Error Recovery Examples

**Query with Fallback Strategy:**
```json
// Primary attempt
{
  "tool": "query_athena_reasoning",
  "parameters": {
    "query": "Complex analysis query",
    "max_reasoning_steps": 5
  }
}

// If timeout, fallback to basic query
{
  "tool": "query_athena", 
  "parameters": {
    "query": "Simplified version of analysis query",
    "mode": "hybrid",
    "top_k": 20
  }
}
```

## Best Practices Summary

### Query Design
1. **Start Simple:** Use basic queries before multi-hop reasoning
2. **Match Mode to Purpose:** Local for specifics, Global for overviews, Hybrid for balance
3. **Optimize Parameters:** Adjust top_k and max_reasoning_steps based on needs
4. **Use Context Strategically:** Choose appropriate context accumulation strategy

### Performance Optimization
1. **Monitor Response Times:** Basic queries <30s, Reasoning <120s
2. **Adjust Parameters:** Lower top_k for faster responses
3. **Cache Common Queries:** Store frequently-used results
4. **Use Appropriate Complexity:** Don't over-engineer simple questions

### Troubleshooting
1. **Check Database Status:** Verify system health first
2. **Test with Simple Queries:** Validate basic functionality
3. **Monitor Resource Usage:** Watch for memory and timeout issues
4. **Use HTTP Mode for Testing:** Debug with curl/Postman when needed

These examples demonstrate the full range of capabilities available through the Athena LightRAG MCP Server, from simple table lookups to complex multi-hop reasoning analysis of healthcare workflows.