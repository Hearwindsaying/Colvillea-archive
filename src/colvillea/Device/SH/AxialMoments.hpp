#pragma once

// Eigen includes
#include <Eigen/Core>


#define AXIAL_EPSILON 1.0E-6

/* _Cosine Sum Integral_
 */
inline __device__ Eigen::VectorXf CosSumIntegral(float x, float y, float c, int n) {

	const float siny  = sin(y);
	const float sinx  = sin(x);
	const float cosy  = cos(y);
	const float cosx  = cos(x);
	const float cosy2 = cosy*cosy;
	const float cosx2 = cosx*cosx;

   const Eigen::Vector2i i1 = {1, 1};
   const Eigen::Vector2i i2 = {2, 2};
   Eigen::Vector2i i = {0, 1};
   Eigen::Vector2f F = {y-x, siny-sinx};
   Eigen::Vector2f S = {0.0f, 0.0f};

   Eigen::VectorXf R = Eigen::VectorXf::Zero(n+2);

   Eigen::Vector2f pow_c    = {1.0, c};
   Eigen::Vector2f pow_cosx = {cosx, cosx2};
   Eigen::Vector2f pow_cosy = {cosy, cosy2};

   while(i[1] <= n) {
      // S <= S + c^{i} F
      S += pow_c.cwiseProduct(F);

      // The resulting vector `R` is shifted of one to the right. This is due to
      // the fact that order `n` moment requires only up to order `n-1` power of
      // cosine integrals.
      R.segment(i[1], 2) = S;

      // T <= cos(y)^{i+1} sin(y) - cos(x)^{i+1} sin(x)
      // F <= (T + i+1) F / (i+2)
      auto T = pow_cosy*siny - pow_cosx*sinx;
      F = (T + (i+i1).cast<float>().cwiseProduct(F)).cwiseQuotient((i+i2).cast<float>());

      // Update temp variable
      i        += i2;
      pow_c    *= c*c;
      pow_cosx *= cosx2;
      pow_cosy *= cosy2;
   }

   return R;
}

/* _sign_
 *
 * Sign function template
 */
template <typename T>
inline __device__ int sign(T val) {
    return (T(0) <= val) - (val < T(0));
}

/* _clamp_
 *
 * Clamp function template to restrict a given function to be in between
 * boundaries.
 */
template <typename T>
inline __device__ T clamp(T val, T a, T b) {
    return max(a, min(b, val));
}


/* _Line Integral_
 */
template<class Vector>
inline __device__ Eigen::VectorXf LineIntegral(const Vector& A, const Vector& B,
                                    const Vector& w, int n) {
#ifdef ZERO_ORTHOGONAL
   auto wDotA = Vector::Dot(A, w);
   auto wDotB = Vector::Dot(B, w);
   // Zeroth moment and orthogonal directions 'w' to the while edge do not
   // require this part since it will always return '0';
   if(std::abs(wDotA) < AXIAL_EPSILON && std::abs(wDotB) < AXIAL_EPSILON) {
      return Eigen::VectorXf::Zero(n+2);
   }
#endif

   // Note: expanding the (I-ssT)B expression from Arvo's LineIntegral pseudo
   // code to the projection of B on plane with A as normal.
   const auto s = Vector::Normalize(A);
   const auto t = Vector::Normalize(B - Vector::Dot(s, B)*s);

   const auto a = Vector::Dot(w, s);
   const auto b = Vector::Dot(w, t);
   const auto c = sqrt(a*a + b*b);

   // Compute the arc-length on which to compute the integral of the moment
   // function and the shift 'p' that change the integral to the integral of
   // a shifted cosine.
   const auto r = Vector::Dot(s, B) / Vector::Dot(B,B);
   const auto l = acos(clamp<float>(r, -1.0f, 1.0f));
   const auto p = atan2(b, a);

   return CosSumIntegral(-p, l-p, c, n);
}

/* _Boundary Integral_
 *
 * Compute the integral along P egdes of the up to order 'n' axial moment
 * around w. By using 'n' = 'w' you can compute the single axis moment. Double
 * axis moment with second axis begin the normal must use 'v' == 'n' ('n' being
 * the normal).
 */
template<class Polygon, class Vector>
inline __device__ Eigen::VectorXf BoundaryIntegral(const Polygon& P, int pSize, const Vector& w,
                                        const Vector& v, int n) {
   // Init to zero
   Eigen::VectorXf b = Eigen::VectorXf::Zero(n+2);

   for (int i = 0; i < pSize; ++i) {
       auto edge = P[i];
      // Compute the edge normal
      const auto normal = Vector::Normalize(Vector::Cross(edge.A, edge.B));

      // Add the egde integral to the total integral
      const auto dotNV   = Vector::Dot(normal, v);
      const auto lineInt = LineIntegral<Vector>(edge.A, edge.B, w, n);
      b += dotNV * lineInt;
   }

   return b;
}

/* _Solid Angle_
 *
 * Compute the solid angle sustained by a `Polygon P`.
 */
template<class Polygon, class Vector>
inline __device__ float SolidAngle(const Polygon& P) {
   if(P.size() == 3) {
      // Using the method of Oosterom and Strackee [1983]
      const Vector& A = P[0].A;
      const Vector& B = P[1].A;
      const Vector& C = P[2].A;

      const Vector bc = Vector::Cross(B,C);
      const float num = std::abs(Vector::Dot(bc, A));
      const float al = Vector::Length(A);
      const float bl = Vector::Length(B);
      const float cl = Vector::Length(C);
      const float den = al*bl*cl
                      + Vector::Dot(A, B)*cl
                      + Vector::Dot(A, C)*bl
                      + Vector::Dot(B, C)*al;

      float phi = atan2(num, den);
      if(phi < 0) {
         phi += M_PI;
      }
      return 2.0f * phi;

   } else {
       rtPrintf("Polygonal SH not supported!");
#if 0

      // Using the algorithm for computing solid angle of polyhedral cones by
      // Mazonka found in http://arxiv.org/pdf/1205.1396v2.pdf
      std::complex<float> z(1, 0);
      for(unsigned int k=0; k<P.size(); ++k) {
         const Vector& A = P[(k > 0) ? k-1 : P.size()-1].A;
         const Vector& B = P[k].A;
         const Vector& C = P[k].B;

         const float ak = Vector::Dot(A, C);
         const float bk = Vector::Dot(A, B);
         const float ck = Vector::Dot(B, C);
         const float dk = Vector::Dot(A, Vector::Cross(B, C));
         const std::complex<float> zk(bk*ck-ak, dk);
         z *= zk;
      }
      const float arg = std::arg(z);
      return arg;
#endif // _DEBUG
   }
}

/* _Check Polygon_
 *
 * Check if the Poylgon P is well oriented. For a triangle, the centroid of the
 * triangle `D` is computed as `A + B + C / 3` and compared to the normal of
 * the triangle using the orientation. The normal and the centroid must match
 * orientation for the normal of edges to be outwards.
 */
template<class Polygon, class Vector>
inline __device__ bool CheckPolygon(const Polygon& P) {

   // A closed Polygon cannot be smaller than 3 Edges.
   if(P.size() < 3) {
      return false;

   // Special case for triangles.
   } else if(P.size() == 3) {
      // Check with respect to centroid
      const auto D = (P[0].A + P[1].A + P[2].A) / 3.0f;
      const auto N = Vector::Cross(P[1].A-P[0].A, P[2].A-P[0].A);
      return Vector::Dot(D, N) <= 0.0f;

   // General computation
   } else {
      // This is a heuristic to select a point on the bounding box of the
      // polygon. The orientation test is then computed on this particular
      // corner.
      unsigned int K = 0;
      const Vector* minX = &P[0].B;
      for(unsigned int k=1; k<P.size(); ++k) {
         const Vector* X = &P[k].B;
         if(X->x < minX->x || (X->x <= minX->x && X->y < minX->y)) {
            minX = X;
            K    = k;
         }
      }

      // Perform the test as defined on the Wikipedia page:
      //   https://en.wikipedia.org/wiki/Curve_orientation
      const int K2 = (K < P.size()-1) ? K+1 : 0;
      const auto D = (P[K].A + P[K].B + P[K2].B) / 3.0f;
      const auto N = Vector::Cross(P[K].B-P[K].A, P[K2].B-P[K].A);
      const bool r = Vector::Dot(D, N) <= 0.0f;
      return r;
   }
}

/* _Axial Moments_
 *
 * input:
 *   + Polygon P: A set of egdes that can be enumerated using iterators.
                  Each edge must enable to access two Vector A and B.
 *   + Vector  w: A 3D vector with elements accessible as x,y,z this
                  vector defines the axis on which to compute moments.
 *   + int     n: The maximum moment order to be computed.
 *
 * output:
 *   + VectorX r: A vector containing all moments up to order 'n'
 */
template<class Polygon, class Vector>
inline __device__ Eigen::VectorXf AxialMoment(const Polygon& P, const Vector& w, int n) {

   // Check if the polygon is well oriented
   const bool check = CheckPolygon<Polygon, Vector>(P);
   if(!check) {
       rtPrintf("not orted well!\n");
      return Eigen::VectorXf::Zero(n+2);
   }

   // Arvo's boundary integral for single vector moment.
   Eigen::VectorXf a = - BoundaryIntegral<Polygon, Vector>(P, P.size(), w, w, n);

   // Generate the 'b' vector which equals to the Polygon solid angle for
   // even moments and zero for odd moments.
   const int n2 = (n+2)/2;
   auto b = Eigen::Map<Eigen::VectorXf, 0, Eigen::InnerStride<2>>(a.data(), n2);
   b += Eigen::VectorXf::Constant(n2, SolidAngle<Polygon, Vector>(P));

   // 'c' is the vector of linear elements, storing 'i+1' for index 'i'
   auto c = Eigen::VectorXf::LinSpaced(n+2, 1, n+2);

   return a.cwiseQuotient(c);
}

/* _Axial Moments_
 *
 * Compute the axial moments for given set of directions used for lobe sharing
 * the maximum cosine order to compute the integral is a function of the size
 * of the directions list.
 */
template<class Polygon, class Vector>
inline __device__ Eigen::VectorXf AxialMoments(const Polygon& P,
                                    const Vector * directions, int dirsSize) {

   const int dsize = dirsSize;
   const int order = (dsize-1) / 2 + 1;

   Eigen::VectorXf result(dsize*order);

   for(int i=0; i<dsize; ++i) {

      // Get the vector associated to the current row
      const Vector& w = directions[i];

      // Evaluate all the Y_{l,m} for the current vector
      const auto In = AxialMoment<Polygon, Vector>(P, w, order-1);

      const int shift = i*order;
      result.segment(shift, order) = In.segment(0, order);
   }
   return result;
}
