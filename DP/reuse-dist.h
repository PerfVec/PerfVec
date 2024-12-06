/*
 * Copyright © 2012, Triad National Security, LLC All rights reserved.
 *
 * This software was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. NEITHER THE GOVERNMENT NOR TRIAD NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
 *
 * Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * Neither the name of Triad National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY TRIAD NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL TRIAD NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

using namespace std;

//typedef CachedUnorderedMap<uint64_t, uint64_t> addr_to_time_t;
typedef unordered_map<uint64_t, uint64_t> addr_to_time_t;

uint64_t bf_max_reuse_distance = 100000000;  // Maximum reuse distance to consider */

// An RDnode is one node in a reuse-distance tree.
class RDnode {
private:
  RDnode* left;         // Left child
  RDnode* right;        // Right child

  // Fix the node's weight (subtree size).
  void fix_node_weight();

  // Fix the weight of all nodes along the path to a given time.
  void fix_path_weights(uint64_t time);

  // Splay a value to the top of the tree, returning the new tree.
  RDnode* splay(uint64_t target);

public:
  uint64_t address;     // Address from trace
  uint64_t time;        // Time of the address's last access
  uint64_t weight;      // Number of items in this subtree (self included)

  // Initialize a new RDnode with a given address and timestamp (defaulting to
  // dummy values).
  RDnode();
  RDnode(uint64_t address, uint64_t time);

  // Reinitialize an existing RDnode with a given address and timestamp.
  void initialize(uint64_t address, uint64_t time);

  // Insert a node into the tree and return the new tree.
  RDnode* insert(RDnode* new_node);

  // Remove a timestamp from the tree and return the new tree and the node that
  // was deleted.
  RDnode* remove(uint64_t timestamp, RDnode** removed_node);

  // Remove all timestamps less than a given value from the tree and from a
  // given histogram, and return the new tree.
  RDnode* prune_tree(uint64_t timestamp, addr_to_time_t* histogram);

  // Return the number of nodes in a splay tree whose timestamp is larger than
  // a given value.
  uint64_t tree_dist(uint64_t timestamp);

  // Ensure that all nodes have a valid weight.
  void validate_weights();
};

// fix_node_weight() sets the weight of a given node to the sum of its
// immediate children's weight plus one.
void RDnode::fix_node_weight()
{
  uint64_t new_weight = 1;
  if (left != nullptr)
    new_weight += left->weight;
  if (right != nullptr)
    new_weight += right->weight;
  weight = new_weight;
}


// fix_path_weights() fixes node weights along the path to a given time.
void RDnode::fix_path_weights(uint64_t target)
{
  // Do an ordinary binary tree search for target -- which we expect not to
  // find -- but change child pointers to parent pointers as we go (instead of
  // requiring extra memory to maintain our path back to the root).
  RDnode* parent = nullptr;
  RDnode* node = this;
  while (node != nullptr) {
    RDnode* child;
    if (target < node->time) {
      child = node->left;
      node->left = parent;
    }
    else {
      child = node->right;
      node->right = parent;
    }
    parent = node;
    node = child;
  }

  // Walk back up the tree, fixing weights and child pointers as we go.
  while (parent != nullptr) {
    RDnode* prev_node = node;
    node = parent;
    if (target < node->time) {
      // We borrowed our left child's pointer.
      parent = node->left;
      node->left = prev_node;
    }
    else {
      // We borrowed our right child's pointer.
      parent = node->right;
      node->right = prev_node;
    }
    node->fix_node_weight();
  }
}


// splay() splays a value (or a nearby value if the value doesn't appear in the
// tree) to the top of the tree, returning the new tree.
RDnode* RDnode::splay(uint64_t target)
{
  RDnode* node = this;
  RDnode new_node;
  new_node.left = nullptr;
  new_node.right = nullptr;
  RDnode* left = &new_node;
  RDnode* right = &new_node;

  while (true) {
    if (target < node->time) {
      if (node->left == nullptr)
        break;
      if (target < node->left->time) {
        // Rotate right
        RDnode* parent = node->left;
        node->left = parent->right;
        parent->right = node;
        node = parent;

        // Fix weights.
        node->right->fix_node_weight();
        node->fix_node_weight();
        if (node->left == nullptr)
          break;
      }

      // Link right
      right->left = node;
      right = node;
      node = node->left;
    }
    else
      if (target > node->time) {
        if (node->right == nullptr)
          break;
        if (target > node->right->time) {
          // Rotate left
          RDnode* parent = node->right;
          node->right = parent->left;
          parent->left = node;
          node = parent;

          // Fix weights.
          node->left->fix_node_weight();
          node->fix_node_weight();
          if (node->right == nullptr)
            break;
        }

        // Link left
        left->right = node;
        left = node;
        node = node->right;
      }
      else
        break;
  }

  // Assemble the final tree.
  left->right = node->left;
  right->left = node->right;
  node->left = new_node.right;
  node->right = new_node.left;

  // Fix weights up to the node from its previous position.
  if (node->left != nullptr)
    node->left->fix_path_weights(node->time);
  if (node->right != nullptr)
    node->right->fix_path_weights(node->time);
  return node;
}

// insert() inserts a new node into a splay tree and returns the new tree.
// Duplicates insertions produce undefined behavior.  Insertions into NULL
// trees produce undefined behavior.  (The caller should check for the
// first-insertion and allocate memory accordingly.)
RDnode* RDnode::insert(RDnode* new_node)
{
  // Handle some simple cases.
  RDnode* node = this;
  node = node->splay(new_node->time);
  if (new_node->time == node->time)
    // The timestamp is already in the tree.  This should never happen when the
    // tree is used for reuse-distance calculations.
    abort();

  // Handle the normal cases.
  if (new_node->time > node->time) {
    new_node->right = node->right;
    new_node->left = node;
    node->right = nullptr;
  }
  else {
    new_node->left = node->left;
    new_node->right = node;
    node->left = nullptr;
  }
  node->fix_node_weight();
  new_node->fix_node_weight();
  return new_node;
}


// remove() deletes a timestamp from the tree and returns the new tree and the
// deleted node.  Missing timestamps produce undefined behavior.
RDnode* RDnode::remove(uint64_t target, RDnode** removed_node)
{
  RDnode* node = this;
  node = node->splay(target);
  if (node->time != target)
    // Not found
    abort();
  RDnode* new_root;
  if (node->left == nullptr)
    // Smallest value in the tree
    new_root = node->right;
  else {
    // Any other value
    new_root = node->left->splay(target);
    if (new_root != nullptr) {
      new_root->right = node->right;
      if (new_root->right != nullptr)
        new_root->right->fix_node_weight();
      new_root->fix_node_weight();
    }
  }
  *removed_node = node;
  return new_root;
}


// Remove all timestamps less than a given value from the tree and from a given
// histogram, and return the new tree and new set of symbols.
RDnode* RDnode::prune_tree(uint64_t timestamp, addr_to_time_t* histogram)
{
  RDnode* new_tree = splay(0);
  while (new_tree && new_tree->time < timestamp) {
    RDnode* dead_node = new_tree;
    new_tree = new_tree->right;
    if (new_tree->left)
      new_tree = new_tree->splay(0);
    histogram->erase(dead_node->address);
    delete dead_node;
  }
  return new_tree;
}


// tree_dist() returns the number of nodes in a splay tree whose timestamp is
// larger than a given value.
uint64_t RDnode::tree_dist(uint64_t timestamp)
{
  RDnode* node = this;
  uint64_t num_larger = 0;
  while (true) {
    if (timestamp > node->time) {
      node = node->right;
    }
    else
      if (timestamp < node->time) {
        num_larger++;
        if (node->right != nullptr)
          num_larger += node->right->weight;
        node = node->left;
      }
      else {
        if (node->right != nullptr)
          num_larger += node->right->weight;
        return num_larger;
      }
  }
}


// For debugging purposes, ensure that every node of a tree contains correct
// weights.
void RDnode::validate_weights()
{
  uint64_t true_weight = 1;
  if (left != nullptr) {
    left->validate_weights();
    true_weight += left->weight;
  }
  if (right != nullptr) {
    right->validate_weights();
    true_weight += right->weight;
  }
  if (weight != true_weight) {
    cerr << "*** Internal error: Node " << this << " has weight "
         << weight << " but expected weight " << true_weight << " ***\n";
    abort();
  }
}

// Reinitialize an existing RDnode with a given address and timestamp.
void RDnode::initialize(uint64_t new_address, uint64_t new_time)
{
  address = new_address;
  time = new_time;
  weight = 1;
  left = nullptr;
  right = nullptr;
}


// Initialize a new RDnode with a dummy address and timestamp.
RDnode::RDnode()
{
  initialize(0, 0);
}


// Initialize a new RDnode with a given address and timestamp.
RDnode::RDnode(uint64_t address, uint64_t time)
{
  initialize(address, time);
}


// Define infinite distance.
const uint64_t infinite_distance = ~(uint64_t)0;


// A ReuseDistance encapsulates all the state needed for a
// reuse-distance calculation.
class ReuseDistance {
private:
  uint64_t clock;           // Current time
  vector<uint64_t> hist;    // Histogram of the number of times each reuse distance was observed
  uint64_t unique_entries;  // Number of unique addresses (infinite reuse distance)
  RDnode* dist_tree;        // Tree of reuse distances
  addr_to_time_t last_access;   // Last access time of a given address

public:
  // Initialize our various fields.
  ReuseDistance() {
    clock = 0;
    unique_entries = 0;
    dist_tree = nullptr;
    last_access.clear();
  }

  // Incorporate a new address into the reuse-distance histogram.
  void process_address(uint64_t address);
  uint64_t process_address(uint64_t address, bool update);

  // Return a pointer to the reuse-distance histogram.
  vector<uint64_t>* get_histogram() { return &hist; }

  // Return the number of unique addresses.
  uint64_t get_unique_addrs() { return unique_entries; }

  // Compute the median reuse distance.
  void compute_median(uint64_t* median_value, uint64_t* mad_value);
};


// Incorporate a new address and output its reuse distance.
uint64_t ReuseDistance::process_address(uint64_t address, bool update)
{
  uint64_t distance = infinite_distance;
  addr_to_time_t::iterator prev_time_iter = last_access.find(address);
  RDnode* new_node = nullptr;
  if (prev_time_iter != last_access.end()) {
    // We've previously seen this address.
    uint64_t prev_time = prev_time_iter->second;
    distance = dist_tree->tree_dist(prev_time) + 1;
    if (update)
      dist_tree = dist_tree->remove(prev_time, &new_node);
  }

  if (update) {
    // Update the tree and the map.
    if (new_node == nullptr)
      new_node = new RDnode(address, clock);
    else
      new_node->initialize(address, clock);
    if (__builtin_expect(dist_tree == nullptr, 0))
      // First insertion into the tree.
      dist_tree = new_node;
    else
      // All other tree insertions.
      dist_tree = dist_tree->insert(new_node);
    last_access[address] = clock;
    clock++;

    // If the tree and the map have grown too large, prune old addresses from
    // them.
    if (last_access.size() > bf_max_reuse_distance)
      dist_tree = dist_tree->prune_tree(clock - bf_max_reuse_distance, &last_access);
  }
  if (distance > bf_max_reuse_distance)
    distance = bf_max_reuse_distance;
  return distance;
}


// Incorporate a new address into the reuse-distance histogram.
void ReuseDistance::process_address(uint64_t address)
{
  // Update the histogram.
  uint64_t distance = infinite_distance;
  addr_to_time_t::iterator prev_time_iter = last_access.find(address);
  RDnode* new_node = nullptr;
  if (prev_time_iter != last_access.end()) {
    // We've previously seen this address.
    uint64_t prev_time = prev_time_iter->second;
    distance = dist_tree->tree_dist(prev_time);
    dist_tree = dist_tree->remove(prev_time, &new_node);
  }
  uint64_t hist_len = hist.size();
  if (distance < hist_len)
    // We've previously seen both this symbol and this reuse distance.
    hist[distance]++;
  else {
    if (distance == infinite_distance)
      // This is the first time we've seen this symbol.
      unique_entries++;
    else {
      // We've previously seen this symbol but not this reuse distance.
      hist.resize(distance+1, 0);
      hist[distance]++;
    }
  }

  // Update the tree and the map.
  if (new_node == nullptr)
    new_node = new RDnode(address, clock);
  else
    new_node->initialize(address, clock);
  if (__builtin_expect(dist_tree == nullptr, 0))
    // First insertion into the tree.
    dist_tree = new_node;
  else
    // All other tree insertions.
    dist_tree = dist_tree->insert(new_node);
  last_access[address] = clock;
  clock++;

  // If the tree and the map have grown too large, prune old addresses from
  // them.
  if (last_access.size() > bf_max_reuse_distance)
    dist_tree = dist_tree->prune_tree(clock - bf_max_reuse_distance, &last_access);
}


// Compute the median reuse distance and the median absolute deviation of that.
void ReuseDistance::compute_median(uint64_t* median_value, uint64_t* mad_value) {
  // Find the total tally.
  uint64_t hist_len = hist.size();   // Entries in the histogram
  uint64_t total_tally;              // Total number of accesses including one-time accesses
  total_tally = unique_entries - hist_len;
  for (size_t dist = 0; dist < hist_len; dist++)
    total_tally += hist[dist];

  // Find the distance that lies at half the total tally.
  uint64_t median_distance = infinite_distance;
  uint64_t median_tally = 0;
  for (size_t dist = 0; dist < hist_len; dist++) {
    median_distance = dist;
    median_tally += hist[dist];
    if (median_tally > total_tally/2)
      break;
  }

  // Tally the absolute deviations.
  vector<uint64_t> absdev(hist_len, 0);
  for (size_t dist = 0; dist < hist_len; dist++) {
    uint64_t tally = hist[dist];
    uint64_t deviation;
    if (dist > median_distance)
      deviation = dist - median_distance;
    else
      deviation = median_distance - dist;
    absdev[deviation] += tally;
  }

  // Find the deviation that lies at half the total tally.
  uint64_t mad = 0;
  uint64_t absdev_tally = 0;
  uint64_t absdev_len = absdev.size();
  for (size_t dev = 0; dev < absdev_len; dev++) {
    mad = dev;
    absdev_tally += absdev[dev];
    if (absdev_tally > total_tally/2)
      break;
  }

  // Return the results.
  *median_value = median_distance;
  *mad_value = mad;
}
