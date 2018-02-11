/*
 * This file is part of the program ltl2dstar (http://www.ltl2dstar.de/).
 * Copyright (C) 2005-2015 Joachim Klein <j.klein@ltl2dstar.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#ifndef HASHFUNCTION_H
#define HASHFUNCTION_H

#include <boost/type_traits/is_integral.hpp>

/** @file
 * Provides HashFunction functors.
 */

/**
 * A hash function using the DJB2 hash function proposed by Dan Bernstein.
 */
class HashFunctionDJB2 {
public:
  /** Constructor. */
  HashFunctionDJB2() :
    _hash(5381) // seeding
  { }

  /** Hash a single char */
  void hash(unsigned char c) {
    HASH(c);
  }

  /** Hash any integral type */
  template <class T>
  void hash(const T v) {
    hash_integral(v, ::boost::is_integral<T>());
  }

  /** Hash a boolean */
  void hash(bool b) {
    HASH((unsigned char)b);
  }

  /** Get current value of the hash */
  unsigned int value() {return _hash;}

private:
  /** Hash a single byte */
  void HASH(unsigned char c) {
    _hash = ((_hash << 5) + _hash) ^ c;    
  }

  template <class T>
  void hash_integral(T v, const boost::true_type& ) {
    for (unsigned int i=0;i<sizeof(T);i++) {
      HASH((unsigned char) v&0xFF);
      v>>=8;
    }
  }

  
  /** The current hash value */
  unsigned int _hash;
};

/** Type of a standard hash function, currently HashFunctionDJB2 */
typedef HashFunctionDJB2 StdHashFunction;



/** A functor using the standard hash function on boost::shared_ptr */
template <typename K >
struct ptr_hash {
  size_t operator()(const K& x) const {
    StdHashFunction hash;
    x->hashCode(hash);
    return hash.value();
  }
};


#endif
