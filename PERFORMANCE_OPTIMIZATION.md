# Settings Page Performance Optimization Guide

## Performance Issues Identified

### 1. Multiple Catalog Loads
**Problem**: `get_cached_catalog()` called 4 times per page load (lines 1764, 1806, 1864, 1869)
**Impact**: ~200-400ms per call if fetching from remote source

### 2. Redundant Database Operations
**Problem**: 
- `storage.init_db()` called twice (lines 1679, 1848)
- `storage.list_providers()` called on every render without caching
**Impact**: ~50-100ms overhead per operation

### 3. Heavy Module-Level Imports
**Problem**: Loading PIL, cryptography, yaml, dotenv on every page load
**Impact**: ~300-500ms import time, ~10-20MB memory

### 4. No Caching Strategy
**Problem**: Expensive operations repeated on every interaction
**Impact**: 2-3 second page loads

### 5. Inefficient Model Processing
**Problem**: Full catalog processed on every filter change
**Impact**: ~100-200ms per interaction

## Optimization Solutions

### 1. Implement Streamlit Caching

```python
# Cache catalog for 1 hour
@st.cache_data(ttl=3600)
def _get_cached_catalog_wrapper():
    return get_cached_catalog()

# Cache database initialization
@st.cache_resource
def _init_database():
    storage.init_db()
    return True

# Cache provider list with short TTL
@st.cache_data(ttl=60)
def _get_providers_list():
    return storage.list_providers()
```

### 2. Lazy Import Pattern

```python
# Cache heavy imports
@st.cache_resource
def _get_heavy_imports():
    import yaml
    from PIL import Image
    from dotenv import load_dotenv
    from cryptography.fernet import Fernet, InvalidToken
    return yaml, Image, load_dotenv, Fernet, InvalidToken

# Use when needed
yaml, Image, load_dotenv, Fernet, InvalidToken = _get_heavy_imports()
```

### 3. Pre-process and Cache Model Lists

```python
@st.cache_data
def _process_catalog_models(catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process catalog once and cache results."""
    all_models = []
    # ... processing logic ...
    return all_models
```

### 4. Session-Level Resource Caching

```python
@st.cache_data
def _get_kms():
    """Cache KMS instance per session."""
    # ... KMS initialization ...
    return fernet_instance
```

## Implementation Steps

### Step 1: Create Optimized Version
1. Copy `2_Settings.py` to `2_Settings_optimized.py`
2. Apply caching decorators
3. Implement lazy loading
4. Consolidate duplicate operations

### Step 2: Test Performance
```bash
# Run benchmark script
python scripts/benchmark_settings.py

# Or test manually
streamlit run app/main.py
```

### Step 3: Gradual Migration
1. Test optimized version alongside original
2. Monitor for any issues
3. Replace original when stable

### Step 4: Apply Pattern to Other Pages
Use similar optimization patterns for:
- `1_Upload_and_Process.py`
- Other heavy pages

## Expected Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Initial Load | 2-3s | 0.5-1s | 60-70% faster |
| Catalog Load | 400ms | 50ms | 87% faster |
| DB Operations | 100ms | 10ms | 90% faster |
| Memory Usage | 50MB | 20MB | 60% reduction |
| Subsequent Loads | 1-2s | 200-400ms | 80% faster |

## Key Optimizations Applied

1. **Lazy Loading**: Defer heavy imports until actually needed
2. **Streamlit Caching**: Use `@st.cache_data` and `@st.cache_resource`
3. **Catalog Processing**: Pre-process and cache model lists
4. **Database Operations**: Single initialization per session
5. **Provider List**: Cache with TTL for reduced DB queries
6. **KMS Instance**: Cache to avoid repeated key loading
7. **Model Processing**: Batch processing with cached results

## Additional Recommendations

### 1. Background Loading
Consider loading catalog in background:
```python
@st.fragment
def load_catalog_async():
    with st.spinner("Loading models..."):
        return _get_cached_catalog_wrapper()
```

### 2. Progressive Enhancement
Show UI immediately, load data progressively:
```python
if 'catalog' not in st.session_state:
    st.session_state.catalog = None
    st.rerun()
```

### 3. Connection Pooling
For database operations:
```python
@st.cache_resource
def get_db_connection():
    return create_engine(url, poolclass=StaticPool)
```

### 4. Client-Side Caching
Use browser caching for static assets:
```python
st.markdown(
    f'<img src="{logo_url}" loading="lazy">',
    unsafe_allow_html=True
)
```

## Monitoring Performance

Add performance metrics:
```python
import time

start = time.time()
# ... operation ...
duration = time.time() - start
st.caption(f"Loaded in {duration:.2f}s")
```

## Rollback Plan

If issues arise:
1. Keep original `2_Settings.py` unchanged
2. Test optimized version as `2_Settings_optimized.py`
3. Switch back by renaming files if needed
4. Monitor error logs and user feedback

## Next Steps

1. Review and test the optimized version
2. Apply similar patterns to other slow pages
3. Consider implementing a global caching strategy
4. Add performance monitoring/metrics
5. Document caching behavior for team