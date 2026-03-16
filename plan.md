# Multi-Course Support Implementation Plan

## Goal
Add a second subject/course to the app with its own File Search store, persona, prompts, data files, and content — while keeping all existing Maritime features intact. The architecture should make adding future courses trivial.

---

## Phase 1: File Search Store Setup (User action — outside code)

**You (the user) will need to:**
1. Create a new Gemini File Search store for the new subject's PDF(s) using `rag_setup_update.py` as a template
2. Note the new store ID (e.g., `fileSearchStores/new-subject-xxxxx`)
3. Add the new store ID to `.streamlit/secrets.toml` as a new key (e.g., `STORE_ID_2 = "fileSearchStores/..."`)

**I will:**
- Create a generalized `rag_setup_new.py` script that accepts a course name, PDF list, and optional existing store ID — so you can run it for any new course

---

## Phase 2: Course Configuration Architecture

Introduce a `CourseConfig` dataclass/dict that bundles all course-specific settings. This is the core abstraction.

### New data structure (added near the top of `claude_demo.py`):

```python
class CourseConfig:
    name: str                    # Display name (e.g., "Maritime Rules of the Road")
    icon: str                    # Emoji icon (e.g., "⚓")
    store_id_secret: str         # Key in st.secrets for this course's File Search store
    system_instruction: str      # RAG-mode system prompt
    system_instruction_open: str # Open LLM mode system prompt
    voice_instruction: str       # Voice chat prompt (RAG)
    voice_instruction_open: str  # Voice chat prompt (Open)
    mc_csv: str                  # Path to multiple-choice CSV
    frq_csv: str                 # Path to free-response CSV
    visual_csv: str              # Path to visual quiz CSV
    background_image: str        # Path to background image (or None)
    rule_knowledge_areas: dict   # Topic mapping for analytics (RULE_KNOWLEDGE_AREAS equivalent)
    app_title: str               # Sidebar title
    version_label: str           # Footer version string
```

### Course registry:
```python
COURSES = {
    "maritime": CourseConfig(name="Maritime Rules of the Road", icon="⚓", ...),
    "new_subject": CourseConfig(name="New Subject Name", icon="📘", ...),
}
```

### Changes to existing code:
- Every place that currently references `STORE_ID_NAME`, `SYSTEM_INSTRUCTION`, `ror_test.csv`, `frq_bank.csv`, `visual_bank.csv`, `RULE_KNOWLEDGE_AREAS`, etc. will instead read from the active `CourseConfig`
- `get_ai_provider()` will become course-aware (or we create one provider per store and switch)
- `GenAIProvider.__init__` already takes `store_id` as a parameter — we just need to pass the right one

---

## Phase 3: Course Switcher UI

### Sidebar addition (above the RAG toggle):
- A `st.selectbox` or styled buttons to pick the active course
- Switching courses will:
  1. Update `st.session_state.active_course`
  2. Clear quiz, FRQ, visual, oral, and voice state (same as "Reset Session" but preserving model settings and analytics)
  3. Trigger `st.rerun()`
- The mode banner, sidebar title, and background will update based on the active course

### Session state changes:
- Add `active_course: str` (key into `COURSES` dict)
- Analytics (`topic_performance`) will be keyed per-course so progress isn't lost when switching: `topic_performance_maritime`, `topic_performance_new_subject`

---

## Phase 4: New Course Content (User provides, I integrate)

### What you need to provide:
1. **The large PDF** — for upload to the new File Search store
2. **System instructions** — describe the AI persona and rules for the new subject
3. **MC questions CSV** — same format as `ror_test.csv` (Question_Text, Option_A-D, Correct_Answer, Meta_Data, Meta_Tags)
4. **FRQ bank CSV** — same format as `frq_bank.csv` (ID, Topic, Question_Text, Ideal_Answer)
5. **Visual quiz images + CSV** — base image, rubric image, and `visual_bank_[course].csv`
6. **Knowledge area mapping** — equivalent of `RULE_KNOWLEDGE_AREAS` for the new subject's topic/chapter structure (for analytics)

### What I will do:
- Create placeholder/template CSVs for the new course
- Wire up the new course config pointing to those files
- Ensure all file paths resolve correctly relative to the course config

---

## Phase 5: Refactor Touchpoints

These are the specific code locations that need to change:

| Current Code | Change |
|---|---|
| `STORE_ID_NAME = "STORE_ID"` | Read from `active_course.store_id_secret` |
| `get_ai_provider()` (cached) | Cache per store_id, or recreate when course changes |
| `SYSTEM_INSTRUCTION` / `SYSTEM_INSTRUCTION_OPEN` | Read from `active_course` |
| `VOICE_CHAT_INSTRUCTION` / `VOICE_CHAT_INSTRUCTION_OPEN` | Read from `active_course` |
| `RULE_KNOWLEDGE_AREAS` dict | Read from `active_course.rule_knowledge_areas` |
| `AnalyticsService` methods | Use course-scoped session state keys |
| `DataManager.load_csv_sample("ror_test.csv", ...)` | Use `active_course.mc_csv` |
| `DataManager.load_frq_bank()` | Use `active_course.frq_csv` |
| `DataManager.load_visual_bank()` | Use `active_course.visual_csv` |
| `set_background()` | Use `active_course.background_image` |
| Sidebar title `"⚓ Maritime AI"` | Use `active_course.icon + active_course.app_title` |
| `validate_config()` | Validate active course's store ID secret |
| Quiz generation prompts (e.g., "Maritime Officer") | Parameterize with course-specific subject terminology |
| Oral Board prompts (e.g., "USCG Licensing Examiner") | Parameterize with course-specific examiner persona |
| Remediation prompts | Parameterize subject references |
| FRQ grading prompt ("Maritime Exam") | Parameterize |

---

## Implementation Order

1. **Create `CourseConfig` model and `COURSES` registry** — define the maritime course with all existing values, plus a skeleton for the new course
2. **Add course switcher to sidebar** — UI + session state wiring
3. **Refactor `get_ai_provider()`** — make it course-aware
4. **Refactor all `get_active_*` helpers** — read from active course config
5. **Refactor `AnalyticsService`** — scope to active course
6. **Refactor all `DataManager` calls** — use course config paths
7. **Parameterize all AI prompts** — replace hardcoded "Maritime" references
8. **Create `rag_setup_new.py`** — generalized store setup script
9. **Create template data files** for the new course
10. **Test** — verify switching between courses preserves all functionality

---

## Questions for You Before We Start

1. **What is the new subject?** (So I can write appropriate system instructions, prompts, and analytics topic mappings)
2. **Do you already have the new File Search store created, or do you need the setup script first?**
3. **Do you have the new CSV content ready (FRQ, MC, Visual), or do you want me to create template files that you'll fill in later?**
4. **Should the new course share the same background image, or will it have its own?**
