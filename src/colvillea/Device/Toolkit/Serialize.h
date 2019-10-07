#pragma once

#ifndef COLVILLEA_DEVICE_TOOLKIT_SERIALIZE_H_
#define COLVILLEA_DEVICE_TOOLKIT_SERIALIZE_H_
#include <ostream>
#include <vector>

/**
 * @brief Serialize a given vector of POD type using STL iostream. Could be used to cache vector to disk and read later to optimize launch time.
 * @param os destination stream
 * @param v vector to be serialized, note that POD type is required.
 * @return the resulted ostream
 */
template<typename POD>
std::ostream& serialize(std::ostream& os, std::vector<POD> const& v)
{
	// this only works on built in data types (PODs)
	static_assert(std::is_trivial<POD>::value && std::is_standard_layout<POD>::value,
		"Can only serialize POD types with this function");

	auto size = v.size();
	os.write(reinterpret_cast<char const*>(&size), sizeof(size));
	os.write(reinterpret_cast<char const*>(v.data()), v.size() * sizeof(POD));
	return os;
}

/**
 * @brief Deserialize a given vector of POD type using STL iostream. Could be used to read vector from disk to optimize launch time.
 * @param is source input stream
 * @param v vector to save for the result vector from input stream
 * @return the resulted istream
 */
template<typename POD>
std::istream& deserialize(std::istream& is, std::vector<POD>& v)
{
	static_assert(std::is_trivial<POD>::value && std::is_standard_layout<POD>::value,
		"Can only deserialize POD types with this function");

	decltype(v.size()) size;
	is.read(reinterpret_cast<char*>(&size), sizeof(size));
	v.resize(size);
	is.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(POD));
	return is;
}

#endif // COLVILLEA_DEVICE_TOOLKIT_SERIALIZE_H_