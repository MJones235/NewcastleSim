# Hybrid Simulation Refactoring Progress

## Overview

Refactoring `hybrid_simulation.py` (1755 lines) into focused, single-responsibility modules.

## Completed Extractions

### 1. PerformanceTimer → `performance_monitor.py` ✅

**Lines:** 78 lines  
**Purpose:** Performance profiling and timing utilities  
**Key Classes:**

- `PerformanceTimer`: Context-manager based timing with parallel operation support

**Reduction:** 1755 → 1686 lines (69 lines removed)

---

### 2. Agent Builder → `agent_builder.py` ✅

**Lines:** 119 lines  
**Purpose:** Agent creation and memory initialization  
**Key Classes:**

- `AgentBuilder`: Builds Concordia agents with memory banks and initial knowledge

**Methods:**

- `build_agents()` - Creates all agents from config
- `_initialize_agent_memory()` - Adds initial memories including station layout

**Reduction:** 1686 → 1629 lines (57 lines removed)  
**Total reduction:** 1755 → 1629 lines (126 lines, 7%)

---

## Pending Extractions

### 3. Observation Builder Module (In Progress)

**Target:** `agent_builder.py`  
**Purpose:** Agent creation and memory initialization  
**Methods to extract:**

- `_build_agents()`
- `_initialize_agent_memory()`
  **Estimated lines:** ~80 lines

### 3. Observation Builder Module (Planned)

**Target:** `observation_builder.py`  
**Purpose:** Generate observations for agents from simulation state  
**Methods to extract:**

- `_generate_observations()`
- `_get_agent_position()`
- `_get_nearby_agents()`
- `_get_recent_events()`
  **Estimated lines:** ~100 lines

### 4. Decision Processor Module (Planned)

**Target:** `decision_processor.py`  
**Purpose:** Agent decision-making and LLM interaction  
**Methods to extract:**

- `_process_agent_decisions()`
- `_process_agents_parallel()`
- `_process_single_agent()`
- `_parse_json_response()`
- `_extract_agent_reasoning()`
  **Estimated lines:** ~400-500 lines (largest module)

### 5. Action Executor Module (Planned)

**Target:** `action_executor.py`  
**Purpose:** Translate and execute actions in JuPedSim  
**Methods to extract:**

- `_apply_action_to_jupedsim()`
- `_convert_speed_to_ms()`
- `_extract_exit_name()`
  **Estimated lines:** ~200 lines

### 6. Message System Module (Planned)

**Target:** `message_system.py`  
**Purpose:** Agent-to-agent communication  
**Methods to extract:**

- `_extract_and_deliver_message()`
- Message memory tracking logic
  **Estimated lines:** ~200 lines

### 7. Helping Coordinator Module (Planned)

**Target:** `helping_coordinator.py`  
**Purpose:** Track and manage help behavior between agents  
**Methods to extract:**

- `_update_helping_relationships()`
- Helper/injured agent coordination logic
  **Estimated lines:** ~150 lines

### 8. Event Manager Module (Planned)

**Target:** `event_manager.py`  
**Purpose:** Event triggering and broadcasting  
**Methods to extract:**

- `_check_and_trigger_events()`
- `block_exit()`
- `broadcast_event()`
  **Estimated lines:** ~100 lines

### 9. Simulation State Module (Planned)

**Target:** `simulation_state.py`  
**Purpose:** Track agent status, exits, destinations  
**Methods to extract:**

- `_check_exited_agents()`
- State tracking data structures
  **Estimated lines:** ~80 lines

### 10. Results/Reporting Module (Planned)

**Target:** `results_manager.py`  
**Purpose:** Save results and generate reports  
**Methods to extract:**

- `save_results()`
- `_save_incremental()`
- `_generate_financial_report()`
  **Estimated lines:** ~250 lines

---

## Final Target

**`hybrid_simulation_runner.py`**

- Slim orchestration layer (~200-300 lines)
- Delegates to focused modules
- Clear main loop
- Minimal state management

**Total Reduction Expected:** 1755 → ~300 lines (83% reduction)

---

## Next Steps

1. Extract Agent Builder
2. Extract Observation Builder
3. Extract Decision Processor (largest module)
4. Extract remaining modules
5. Update imports and test after each extraction

## Testing Strategy

- Compile check after each extraction
- Run simulation to verify behavior unchanged
- Git commit after each successful extraction
