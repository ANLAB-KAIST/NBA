#ifndef __NBA_NODELOCALSTORAGE_HH__
#define __NBA_NODELOCALSTORAGE_HH__

#include "log.hh"

#include <rte_memory.h>
#include <rte_malloc.h>
#include <rte_rwlock.h>
#include <rte_spinlock.h>

#include <unordered_map>
#include <string>

namespace nba {

class NodeLocalStorage {
    /**
     * NodeLocalStorage is a node-version of thread local storage.
     * Elements can use this to store per-node information, such as
     * routing tables.  The storage unit is mapped with a string key
     * for convenience.  The keys from different elements must not
     * overlap as this implementation does not have separate namespaces
     * for each element instances.
     *
     * The elements must use the methods of this class only on
     * initialization steps, and must NOT use them in the data-path.
     * They must keep the pointers and rwlocks retrieved during
     * initialization step to use them in the data-path.
     * It is optional for elements to use rwlocks.  It is recommended
     * to use them only if the storage needs to be updated inside the
     * data-path.
     * TODO: add multiple types of storage, such as RCU-backed ones.
     *
     * alloc() method must be called inside initialize_per_node() method
     * of elements, and get_alloc() / get_rwlock() methods should be
     * called inside configure() method which is called per thread ( =
     * per element instance).
     */
public:
    NodeLocalStorage(unsigned node_id)
    {
        _node_id = node_id;
        for (int i = 0; i < NBA_MAX_NODELOCALSTORAGE_ENTRIES; i++) {
            _pointers[i] = NULL;
            //_rwlocks[i] = NULL;
        }
        rte_spinlock_init(&_node_lock);
    }

    virtual ~NodeLocalStorage()
    {
        // TODO: free all existing entries.
    }

    int alloc(const char *key, size_t size)
    {
        rte_spinlock_lock(&_node_lock);
        size_t kid = _keys.size();
        assert(kid < NBA_MAX_NODELOCALSTORAGE_ENTRIES);
        _keys.insert(std::pair<std::string, int>(key, kid));

        void *ptr = rte_malloc_socket("nls_alloc", size, CACHE_LINE_SIZE, _node_id);
        //void *ptr = new char*[size];
        assert(ptr != NULL);
        memset(ptr, 0xcd, size);
        size_t real_size = 0;
        //assert(0 == rte_malloc_validate(ptr, &real_size));
        _pointers[kid] = ptr;
        RTE_LOG(DEBUG, ELEM, "NLS[%u]: malloc req size %'lu bytes, real size %'lu bytes\n", _node_id, size, real_size);

        //rte_rwlock_t *rwlock = (rte_rwlock_t *) rte_malloc_socket("nls_lock", sizeof(rte_rwlock_t), 64, _node_id);
        //assert(rwlock != NULL);
        //rte_rwlock_init(rwlock);
        //_rwlocks[kid] = rwlock;

        rte_spinlock_unlock(&_node_lock);
        return kid;
    }

    void* get_alloc(const char *key)
    {
        rte_spinlock_lock(&_node_lock);
        assert(_keys.find(key) != _keys.end());
        int kid = _keys[key];
        void *ptr = _pointers[kid];
        rte_spinlock_unlock(&_node_lock);
        return ptr;
    }

    rte_rwlock_t *get_rwlock(const char *key)
    {
        rte_spinlock_lock(&_node_lock);
        assert(_keys.find(key) != _keys.end());
        int kid = _keys[key];
        //rte_rwlock_t *lock = _rwlocks[kid];
        rte_spinlock_unlock(&_node_lock);
        //return lock;
        return nullptr;
    }

    void free(const char *key)
    {
        rte_spinlock_lock(&_node_lock);
        assert(_keys.find(key) != _keys.end());
        int kid = _keys[key];
        void *ptr = _pointers[kid];
        rte_free(ptr);
        //delete (char*)ptr;
        //rte_rwlock_t *rwlock = _rwlocks[kid];
        //rte_free(rwlock);
        _pointers[kid] = NULL;
        rte_spinlock_unlock(&_node_lock);
        // TODO: remove entry from _pointers, _rwlocks, and _keys.
        // But we do not implement it because usually node-local
        // storage is alloccated-once and used-forever.
    }

protected:
    unsigned _node_id;
    //rte_rwlock_t *_rwlocks[NBA_MAX_NODELOCALSTORAGE_ENTRIES];
    void *_pointers[NBA_MAX_NODELOCALSTORAGE_ENTRIES];
    std::unordered_map<std::string, int> _keys;
    rte_spinlock_t _node_lock;
};
}

#endif

// vim: ts=8 sts=4 sw=4 et
