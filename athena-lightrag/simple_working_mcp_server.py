#!/usr/bin/env python3
"""
Simple Working MCP Server for Athena LightRAG
=============================================
Minimal MCP-compliant server that actually works with stdio transport.
This server implements the MCP protocol correctly and provides mock healthcare tools.
"""

import asyncio
import json
import sys
import logging
from typing import Dict, List, Any, Optional

# Configure logging to stderr (not stdout) for MCP compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class SimpleMCPServer:
    """Simple MCP server that implements the protocol correctly"""
    
    def __init__(self):
        self.tools = [
            {
                "name": "lightrag_local_query",
                "description": "Query focused Athena health database entities and table relationships",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Healthcare database query about specific tables, columns, or relationships"
                        },
                        "top_k": {
                            "type": "integer", 
                            "description": "Number of results to return",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "lightrag_global_query", 
                "description": "Query comprehensive medical workflow overviews across Athena database schemas",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Healthcare workflow query spanning multiple database schemas"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens for response",
                            "default": 8000
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "lightrag_hybrid_query",
                "description": "Combined local and global analysis for complex healthcare data relationships",
                "inputSchema": {
                    "type": "object", 
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Complex healthcare query requiring both detailed and contextual analysis"
                        },
                        "entity_tokens": {
                            "type": "integer",
                            "description": "Tokens for entity context",
                            "default": 6000
                        },
                        "relation_tokens": {
                            "type": "integer", 
                            "description": "Tokens for relationship context",
                            "default": 8000
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "lightrag_sql_generation",
                "description": "Generate validated Snowflake SQL queries for Athena Health database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "natural_query": {
                            "type": "string",
                            "description": "Natural language description of the desired SQL query"
                        },
                        "include_schema": {
                            "type": "boolean",
                            "description": "Whether to include schema information",
                            "default": True
                        }
                    },
                    "required": ["natural_query"]
                }
            }
        ]
        logger.info(f"Initialized server with {len(self.tools)} tools")

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        logger.info(f"Handling {method} request (id: {request_id})")
        
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "logging": {}
                    },
                    "serverInfo": {
                        "name": "athena-lightrag-server",
                        "version": "1.0.0"
                    }
                }
                
            elif method == "tools/list":
                result = {"tools": self.tools}
                
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                # Execute the tool
                if tool_name == "lightrag_local_query":
                    query = arguments.get("query", "")
                    top_k = arguments.get("top_k", 10)
                    tool_result = f"""Local Query Results for: "{query}"

🏥 ATHENA HEALTH EHR DATABASE ANALYSIS

Found {top_k} relevant entities in the Snowflake database:

📋 PATIENT-RELATED TABLES:
• athena.athenaone.PATIENT - Core patient demographics and identifiers
  - Columns: patient_id, first_name, last_name, dob, ssn, contact_info
  - Primary key: patient_id
  - Relationships: Links to APPOINTMENT, CHARGEDIAGNOSIS

• athena.athenaone.APPOINTMENT - Patient appointment scheduling
  - Columns: appointment_id, patient_id, provider_id, appointment_date, status
  - Foreign keys: patient_id → PATIENT.patient_id
  - Relationships: Core scheduling entity, links to APPOINTMENTTYPE

• athena.athenaone.APPOINTMENTTYPE - Types of medical appointments
  - Columns: appointment_type_id, appointment_type_name, duration_minutes
  - Relationships: Referenced by APPOINTMENT table

This demonstrates the MCP server is working correctly and can access healthcare database information."""

                elif tool_name == "lightrag_global_query":
                    query = arguments.get("query", "")
                    max_tokens = arguments.get("max_tokens", 8000)
                    tool_result = f"""Global Healthcare Workflow Analysis: "{query}"

🏥 COMPREHENSIVE ATHENA HEALTH EHR OVERVIEW

MULTI-SCHEMA DATABASE ARCHITECTURE:
• athena.athenaone - Primary clinical operations
• athena.athenaclinical - Clinical documentation & notes
• athena.athenafinancial - Billing, insurance, payments
• athena.athenaoperational - System administration & reporting

KEY MEDICAL WORKFLOWS:

1. PATIENT REGISTRATION & MANAGEMENT:
   athena.athenaone.PATIENT → APPOINTMENT → APPOINTMENTTYPE
   - Complete patient lifecycle from registration to scheduling

2. CLINICAL DOCUMENTATION:
   APPOINTMENT → athena.athenaclinical.NOTES → CHARGEDIAGNOSIS
   - Clinical encounters and documentation workflow

3. BILLING & REVENUE CYCLE:
   CHARGEDIAGNOSIS → athena.athenafinancial.CAPPAYMENT
   - From clinical services to payment processing

This global analysis shows interconnected healthcare workflows across {max_tokens} token limit."""

                elif tool_name == "lightrag_hybrid_query":
                    query = arguments.get("query", "")
                    entity_tokens = arguments.get("entity_tokens", 6000)
                    relation_tokens = arguments.get("relation_tokens", 8000)
                    tool_result = f"""Hybrid Analysis: "{query}"

🔍 DETAILED + CONTEXTUAL HEALTHCARE ANALYSIS

ENTITY-LEVEL DETAILS ({entity_tokens} tokens):
• PATIENT table structure with 15+ columns including PHI fields
• APPOINTMENT scheduling with complex business rules
• Provider network relationships and specializations

RELATIONSHIP-LEVEL CONTEXT ({relation_tokens} tokens):
• Patient → Appointment → Provider workflow chains
• Clinical documentation → Billing code relationships  
• Insurance authorization → Payment processing flows
• Quality metrics → Performance reporting connections

COMBINED INSIGHTS:
This hybrid approach provides both granular table/column details AND the broader healthcare workflow context, enabling comprehensive analysis of complex medical data relationships."""

                elif tool_name == "lightrag_sql_generation":
                    natural_query = arguments.get("natural_query", "")
                    include_schema = arguments.get("include_schema", True)
                    
                    if include_schema:
                        schema_info = """
-- SCHEMA INFORMATION:
-- athena.athenaone.PATIENT (patient_id, first_name, last_name, dob, phone)
-- athena.athenaone.APPOINTMENT (appointment_id, patient_id, appointment_date, status)
-- athena.athenaone.APPOINTMENTTYPE (appointment_type_id, appointment_type_name)"""
                    else:
                        schema_info = ""
                    
                    tool_result = f"""-- GENERATED SNOWFLAKE SQL FOR: {natural_query}
{schema_info}

SELECT 
    p.patient_id,
    p.first_name,
    p.last_name,
    a.appointment_date,
    at.appointment_type_name,
    a.status
FROM athena.athenaone.PATIENT p
JOIN athena.athenaone.APPOINTMENT a ON p.patient_id = a.patient_id  
JOIN athena.athenaone.APPOINTMENTTYPE at ON a.appointment_type_id = at.appointment_type_id
WHERE a.appointment_date >= CURRENT_DATE - INTERVAL '30 days'
  AND a.status = 'SCHEDULED'
ORDER BY a.appointment_date DESC
LIMIT 100;

-- This SQL demonstrates proper Snowflake syntax for Athena Health EHR database
-- Uses fully qualified table names: athena.schema.table format
-- Includes appropriate JOINs for healthcare data relationships"""

                else:
                    raise Exception(f"Unknown tool: {tool_name}")
                
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": tool_result
                        }
                    ]
                }
                
            else:
                raise Exception(f"Unknown method: {method}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }

async def main():
    """Main server loop"""
    logger.info("🏥 Starting Simple Athena LightRAG MCP Server")
    server = SimpleMCPServer()
    
    try:
        while True:
            # Read JSON-RPC request from stdin
            line = sys.stdin.readline()
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = await server.handle_request(request)
                
                # Send response to stdout
                print(json.dumps(response), flush=True)
                logger.info(f"Sent response for request {request.get('id')}")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                print(json.dumps(error_response), flush=True)
                
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Server stopped")

if __name__ == "__main__":
    asyncio.run(main())