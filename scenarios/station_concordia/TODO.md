# TODO List for Station Concordia

## ✅ COMPLETED - MVP Phase 1 (Core Framework)

### Initial Setup
- [x] Feasibility assessment completed (HIGH viability)
- [x] Project structure created (15 files, ~2,650 lines)
- [x] Core components implemented (agent, game master, simulation runner)
- [x] Mock JuPedSim for testing
- [x] Configuration system with YAML
- [x] Comprehensive documentation (~15,000 words)

### MVP Validation
- [x] **First successful simulation run completed!**
  - 1 agent evacuating to north exit
  - Decisions logged every 5 seconds
  - Action translation working (NL → coordinates)
  - Mock movement simulation functional
  - Output: `scenarios/station_concordia/output/agent_decisions.json`

### Technical Fixes Applied
- [x] Fixed import: concordia.typing.entity (not concordia.agents.entity)
- [x] Fixed file corruption in hybrid_simulation.py
- [x] Forced CPU for sentence-transformers (GTX 980 CUDA compatibility)
- [x] Fixed --no-llm flag with proper early return
- [x] Fixed MockModel signature to accept **kwargs

### Azure LLM Integration
- [x] **FIXED! Azure LLM now working with Concordia**
  - Created `AzureLLMConcordia` class with synchronous REST API calls
  - Avoids all async/sync event loop conflicts
  - Uses `max_completion_tokens` for newer Azure models
  - Full retry logic and error handling
  - Generates rich, contextual evacuation decisions
  - See: `AZURE_LLM_SUCCESS.md` for details

### Remaining Limitations (MVP)
- ⚠️ Using mock JuPedSim instead of real pedestrian simulation
- ⚠️ Single agent only (hardcoded MVP limit)
- ⚠️ Simplified event system (2 events only)

## Critical TODOs for Phase 2 (JuPedSim Integration)

### core/mock_jupedsim.py
- [ ] **TODO: Replace entire file with real JuPedSim integration**
- [ ] **TODO: See scenarios/station_jupedsim/core/simulation.py for reference**

### core/hybrid_simulation.py
- [ ] **TODO: Verify _step_jupedsim() works with real JuPedSim**
- [ ] **TODO: Verify _get_agent_position() works with real JuPedSim**
- [ ] **TODO: Verify _get_nearby_agents() works with real JuPedSim**
- [ ] **TODO: With real JuPedSim, use proper waypoint/goal setting API in _apply_action_to_jupedsim()**
- [ ] **TODO: Implement _check_and_trigger_events() with time-based event triggering**

### run_station_concordia.py
- [ ] **TODO: Replace MockJuPedSimulation with real JuPedSim**
- [ ] **TODO: Load actual station geometry from scenarios/station_sim/network**
- [ ] **TODO: Use proper spawn points from geometry instead of hardcoded (50, 50)**

## Medium Priority (Phase 3 - Events)

### Event System Integration
- [ ] Connect to scenarios.station_jupedsim.core.event_system.EventManager
- [ ] Implement dynamic event detection based on crowd density
- [ ] Add agent-to-agent messaging
- [ ] Implement event configuration loading from YAML

## Lower Priority (Phase 4-5)

### Performance Optimization
- [ ] Implement async batch processing of LLM calls
- [ ] Add decision caching with LRU cache
- [ ] Implement per-agent cooldown tracking
- [ ] Profile performance bottlenecks

### Advanced Features
- [ ] Add planning component for multi-step evacuation plans
- [ ] Implement group formation dynamics
- [ ] Add emotional state modeling (panic, calm)
- [ ] Create custom visualization integration

### Testing
- [ ] Create unit tests for all components
- [ ] Add integration tests for full simulation
- [ ] Create performance benchmarks
- [ ] Add regression tests

## Dependencies Required

### For Current MVP
- [x] gdm-concordia (pip install gdm-concordia)
- [x] sentence-transformers (pip install sentence-transformers)
- [x] pyyaml (pip install pyyaml)

### For Real JuPedSim Integration
- [ ] jupedsim
- [ ] shapely
- [ ] lxml (for geometry loading)

## Testing Checklist

### MVP (Mock JuPedSim)
- [x] Configuration loads correctly
- [ ] Concordia agents can be created
- [ ] Mock simulation runs end-to-end
- [ ] Agent decisions are logged
- [ ] Actions are translated correctly
- [ ] Observations are generated

### With Real JuPedSim
- [ ] JuPedSim simulation initializes
- [ ] Agents spawn at correct positions
- [ ] Position queries work
- [ ] Spatial queries (nearby agents) work
- [ ] Waypoint setting works
- [ ] Movement looks realistic

## Notes

- All TODOs marked with **TODO:** in code comments
- Mock components clearly labeled for replacement
- Each TODO includes reference to real implementation when available
- Keep this file updated as TODOs are completed

---

**Last Updated:** 2026-02-02
**Current Phase:** MVP with Mock JuPedSim
**Next Milestone:** Replace mock with real JuPedSim
