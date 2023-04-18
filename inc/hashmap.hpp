#ifndef __HASHMAP_H
#define __HASHMAP_H

template <typename K, typename V> struct PairNode {
    K key;
    V value;
    PairNode* next;
};

template <typename K, typename V, typename F> class HashMap {
  public:
    PairNode<K, V>** table;
    int table_size;
    F hash_function;
    HashMap(int table_size) {
        this->table_size = table_size;
        table = new PairNode<K, V>*[this->table_size]();
    }
    ~HashMap() {
        for (int i = 0; i < this->table_size; ++i) {
            PairNode<K, V>* entry = table[i];
            while (entry != nullptr) {
                PairNode<K, V>* prev = entry;
                entry = entry->next;
                delete prev;
            }
            table[i] = nullptr;
        }
        delete[] table;
    }

    bool get(const K &key, PairNode<K, V> * &value) {
        unsigned long hash_value = hash_function(key);
        PairNode<K, V> *entry = table[hash_value];

        while (entry != nullptr) {
            if (entry->key == key) {
                value = entry;
                return true;
            }
            entry = entry->next;
        }
        return false;
    }

    void put(const K& key, const V& value) {
        unsigned long hash_value = hash_function(key);
        PairNode<K, V>* prev = nullptr;
        PairNode<K, V>* entry = table[hash_value];

        while (entry != nullptr && entry->key != key) {
            prev = entry;
            entry = entry->next;
        }

        if (entry == nullptr) {
            entry = new PairNode<K, V>(key, value);
            if (prev == nullptr) {
                table[hash_value] = entry;
            } else {
                prev->next = entry;
            }
        } else {
            entry->value = value;
        }
    }
};

#endif