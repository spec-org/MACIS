#pragma once
#include "sd_operations.hpp"
#include "cmz_ed/lobpcg_call.h++"
#include <ips4o.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>

namespace dbwy {


template <size_t N>
struct asci_contrib {
  std::bitset<N> state;
  double         rv;
};

template <size_t N>
void reorder_ci_on_coeff( std::vector<std::bitset<N>>& dets, 
  std::vector<double>& C_local, MPI_Comm /* comm: will need for dist*/ ) {

  size_t nlocal = C_local.size();
  size_t ndets  = dets.size();
  std::vector<uint64_t> idx( nlocal );
  std::iota( idx.begin(), idx.end(), 0 );
  std::sort( idx.begin(), idx.end(), [&](auto i, auto j) {
    return std::abs(C_local[i]) > std::abs(C_local[j]);
  });

  std::vector<double> reorder_C( nlocal );
  std::vector<std::bitset<N>> reorder_dets( ndets );
  assert( nlocal == ndets );
  for( auto i = 0ul; i < ndets; ++i ) {
    reorder_C[i]    = C_local[idx[i]];
    reorder_dets[i] = dets[idx[i]];
  }

  C_local = std::move(reorder_C);
  dets    = std::move(reorder_dets);

}

template <size_t N, size_t NShift>
void append_singles_asci_contributions( 
  double                       coeff,
  std::bitset<2*N>             state_full,
  std::bitset<N>               state_same,
  const std::vector<uint32_t>& occ_same,
  const std::vector<uint32_t>& vir_same,
  const std::vector<uint32_t>& occ_othr,
  const double*                T_pq,
  const double*                G_kpq,
  const double*                V_kpq,
  size_t                       norb,
  double                       h_el_tol,
  std::vector< asci_contrib<2*N> >& asci_contributions
) {

  const std::bitset<2*N> one = 1;
  const size_t norb2 = norb*norb;
  for( auto i : occ_same )
  for( auto a : vir_same ) {

    double h_el = T_pq[a + i*norb];

    const double* G_ov = G_kpq + a*norb + i*norb2;
    const double* V_ov = V_kpq + a*norb + i*norb2;
    for( auto p : occ_same ) h_el += G_ov[p];
    for( auto p : occ_othr ) h_el += V_ov[p];

    // Early Exit
    if( std::abs(h_el) < h_el_tol ) continue;

    // Calculate Excited Determinant
    auto ex_det = state_full ^ (one << (i+NShift)) ^ (one << (a+NShift));

    // Calculate Excitation Sign in a Canonical Way
    auto sign = single_excitation_sign( state_same, a, i );
    h_el *= sign;

    // Append to return values
    asci_contributions.push_back( {ex_det, coeff*h_el} );

  } // Loop over single extitations

}

template <size_t N, size_t NShift>
void append_singles_asci_contributions( 
  double                       coeff,
  std::bitset<2*N>             state_full,
  std::bitset<N>               state_same,
  const std::vector<uint32_t>& occ_same,
  const std::vector<uint32_t>& vir_same,
  const std::vector<uint32_t>& occ_othr,
  const double*                T_pq,
  const double*                G_kpq,
  const double*                V_kpq,
  size_t                       norb,
  double                       h_el_tol,
  double                       hstate,
  double                       Egs,
  HamiltonianGenerator<2*N>&   ham_gen,
  std::vector< asci_contrib<2*N> >& asci_contributions
) {

  const std::bitset<2*N> one = 1;
  const size_t norb2 = norb*norb;
  for( auto i : occ_same )
  for( auto a : vir_same ) {

    double h_el = T_pq[a + i*norb];

    const double* G_ov = G_kpq + a*norb + i*norb2;
    const double* V_ov = V_kpq + a*norb + i*norb2;
    for( auto p : occ_same ) h_el += G_ov[p];
    for( auto p : occ_othr ) h_el += V_ov[p];

    // Early Exit
    if( std::abs(h_el) < h_el_tol ) continue;

    // Calculate Excited Determinant
    auto ex_det = state_full ^ (one << (i+NShift)) ^ (one << (a+NShift));

    // Calculate Excitation Sign in a Canonical Way
    auto sign = single_excitation_sign( state_same, a, i );
    h_el *= sign;
 
    // Calculate fast diagonal matrix element
    auto hdiag = ham_gen.fast_diag_single( occ_same, occ_othr, i, a, hstate );
    h_el /= ( Egs - hdiag );

    // Append to return values
    asci_contributions.push_back( {ex_det, coeff*h_el} );

  } // Loop over single extitations

}

template <size_t N, size_t NShift>
void append_ss_doubles_asci_contributions( 
  double                       coeff,
  std::bitset<2*N>             state_full,
  std::bitset<N>               state_spin,
  const std::vector<uint32_t>& occ,
  const std::vector<uint32_t>& vir,
  const double*                G,
  size_t                       norb,
  double                       h_el_tol,
  std::vector< asci_contrib<2*N> >& asci_contributions
) {

  const size_t nocc = occ.size();
  const size_t nvir = vir.size();

  const std::bitset<N> one = 1;
  for( auto ii = 0; ii < nocc; ++ii )
  for( auto aa = 0; aa < nvir; ++aa ) {

    const auto i = occ[ii];
    const auto a = vir[aa];
    const auto G_ai = G + (a + i*norb)*norb*norb;

    for( auto jj = ii + 1; jj < nocc; ++jj )
    for( auto bb = aa + 1; bb < nvir; ++bb ) {
      const auto j = occ[jj];
      const auto b = vir[bb];
      const auto jb = b + j*norb;
      const auto G_aibj = G_ai[jb];

      if( std::abs(G_aibj) < h_el_tol ) continue;

      // Calculate excited determinant string (spin)
      const auto full_ex_spin = (one << i) ^ (one << j) ^ (one << a) ^ (one << b);
      auto ex_det_spin = state_spin ^ full_ex_spin;

      // Calculate the sign in a canonical way
      double sign = doubles_sign( state_spin, ex_det_spin, full_ex_spin );

      // Calculate full excited determinant
      const auto full_ex = expand_bitset<2*N>(full_ex_spin) << NShift;
      auto ex_det = state_full ^ full_ex;

      // Update sign of matrix element
      auto h_el = sign * G_aibj;

      // Append {det, c*h_el}
      asci_contributions.push_back( {ex_det, coeff*h_el} );

    } // Restricted BJ loop
  } // AI Loop


}

template <size_t N, size_t NShift>
void append_ss_doubles_asci_contributions( 
  double                       coeff,
  std::bitset<2*N>             state_full,
  std::bitset<N>               state_spin,
  const std::vector<uint32_t>& ss_occ,
  const std::vector<uint32_t>& vir,
  const std::vector<uint32_t>& os_occ,
  const double*                G,
  size_t                       norb,
  double                       h_el_tol,
  double                       hstate,
  double                       Egs,
  HamiltonianGenerator<2*N>&   ham_gen,
  std::vector< asci_contrib<2*N> >& asci_contributions
) {

  const size_t nocc = ss_occ.size();
  const size_t nvir = vir.size();

  const std::bitset<N> one = 1;
  for( auto ii = 0; ii < nocc; ++ii )
  for( auto aa = 0; aa < nvir; ++aa ) {

    const auto i = ss_occ[ii];
    const auto a = vir[aa];
    const auto G_ai = G + (a + i*norb)*norb*norb;

    for( auto jj = ii + 1; jj < nocc; ++jj )
    for( auto bb = aa + 1; bb < nvir; ++bb ) {
      const auto j = ss_occ[jj];
      const auto b = vir[bb];
      const auto jb = b + j*norb;
      const auto G_aibj = G_ai[jb];

      if( std::abs(G_aibj) < h_el_tol ) continue;

      // Calculate excited determinant string (spin)
      const auto full_ex_spin = (one << i) ^ (one << j) ^ (one << a) ^ (one << b);
      auto ex_det_spin = state_spin ^ full_ex_spin;

      // Calculate the sign in a canonical way
      double sign = doubles_sign( state_spin, ex_det_spin, full_ex_spin );

      // Calculate full excited determinant
      const auto full_ex = expand_bitset<2*N>(full_ex_spin) << NShift;
      auto ex_det = state_full ^ full_ex;

      // Update sign of matrix element
      auto h_el = sign * G_aibj;

      // Evaluate fast diagonal matrix element
      auto hdiag = ham_gen.fast_diag_ss_double( ss_occ, os_occ, i, j, a, b, hstate );
      h_el /= (Egs - hdiag);

      // Append {det, c*h_el}
      asci_contributions.push_back( {ex_det, coeff*h_el} );

    } // Restricted BJ loop
  } // AI Loop


}


template <size_t N>
void append_os_doubles_asci_contributions( 
  double                       coeff,
  std::bitset<2*N>             state_full,
  std::bitset<N>               state_alpha,
  std::bitset<N>               state_beta,
  const std::vector<uint32_t>& occ_alpha,
  const std::vector<uint32_t>& occ_beta,
  const std::vector<uint32_t>& vir_alpha,
  const std::vector<uint32_t>& vir_beta,
  const double*                V,
  size_t                       norb,
  double                       h_el_tol,
  std::vector< asci_contrib<2*N> >& asci_contributions
) {

  const std::bitset<2*N> one = 1;
  for( auto i : occ_alpha )
  for( auto a : vir_alpha ) {
    const auto V_ai = V + a + i*norb;

    double sign_alpha = single_excitation_sign( state_alpha, a, i );
    for( auto j : occ_beta )
    for( auto b : vir_beta ) {
      const auto jb = b + j*norb;
      const auto V_aibj = V_ai[jb*norb*norb];

      if( std::abs(V_aibj) < h_el_tol ) continue;

      double sign_beta = single_excitation_sign( state_beta,  b, j );
      double sign = sign_alpha * sign_beta;
      auto ex_det = state_full ^ (one << i) ^ (one << a) ^
                               (((one << j) ^ (one << b)) << N);
      auto h_el = sign * V_aibj;

      asci_contributions.push_back( {ex_det, coeff*h_el} );
    }
  }

}

template <size_t N>
void append_os_doubles_asci_contributions( 
  double                       coeff,
  std::bitset<2*N>             state_full,
  std::bitset<N>               state_alpha,
  std::bitset<N>               state_beta,
  const std::vector<uint32_t>& occ_alpha,
  const std::vector<uint32_t>& occ_beta,
  const std::vector<uint32_t>& vir_alpha,
  const std::vector<uint32_t>& vir_beta,
  const double*                V,
  size_t                       norb,
  double                       h_el_tol,
  double                       hstate,
  double                       Egs,
  HamiltonianGenerator<2*N>&   ham_gen,
  std::vector< asci_contrib<2*N> >& asci_contributions
) {

  const std::bitset<2*N> one = 1;
  for( auto i : occ_alpha )
  for( auto a : vir_alpha ) {
    const auto V_ai = V + a + i*norb;

    double sign_alpha = single_excitation_sign( state_alpha, a, i );
    for( auto j : occ_beta )
    for( auto b : vir_beta ) {
      const auto jb = b + j*norb;
      const auto V_aibj = V_ai[jb*norb*norb];

      if( std::abs(V_aibj) < h_el_tol ) continue;

      double sign_beta = single_excitation_sign( state_beta,  b, j );
      double sign = sign_alpha * sign_beta;
      auto ex_det = state_full ^ (one << i) ^ (one << a) ^
                               (((one << j) ^ (one << b)) << N);
      auto h_el = sign * V_aibj;

      // Evaluate fast diagonal element
      auto hdiag = ham_gen.fast_diag_os_double( occ_alpha, occ_beta, i, j, a, b, hstate );
      h_el /= ( Egs - hdiag );

      asci_contributions.push_back( {ex_det, coeff*h_el} );
    }
  }

}



template <size_t N>
void sort_and_accumulate_asci_pairs(
  std::vector< asci_contrib<N> >& asci_pairs
) {

  // Sort by bitstring
  #if 0
  std::sort( asci_pairs.begin(), asci_pairs.end(), []( auto x, auto y ) {
    return bitset_less(x.state, y.state);
  });
  #else
  #if 0
  ips4o::sort( asci_pairs.begin(), asci_pairs.end(), []( auto x, auto y ) {
    return bitset_less(x.state, y.state);
  });
  #else
  ips4o::parallel::sort( asci_pairs.begin(), asci_pairs.end(), 
    []( auto x, auto y ) { return bitset_less(x.state, y.state); });
  #endif
  #endif

  // Accumulate the ASCI scores into first instance of unique bitstrings
  auto cur_it = asci_pairs.begin();
  for( auto it = cur_it + 1; it != asci_pairs.end(); ++it ) {
    // If iterate is not the one being tracked, update the iterator
    if( it->state != cur_it->state ) { cur_it = it; }

    // Accumulate
    else {
      cur_it->rv += it->rv;
      it->rv = 0; // Zero out to expose potential bugs
    }
  }

  // Remote duplicate bitstrings
  auto uit = std::unique( asci_pairs.begin(), asci_pairs.end(),
    [](auto x, auto y){ return x.state == y.state; } );
  asci_pairs.erase(uit, asci_pairs.end()); // Erase dead space

}

template <size_t N>
void keep_only_largest_copy_asci_pairs(
  std::vector< asci_contrib<N> >& asci_pairs
) {

  // Sort by bitstring
  #if 0
  std::sort( asci_pairs.begin(), asci_pairs.end(), []( auto x, auto y ) {
    return bitset_less(x.state, y.state);
  });
  #else
  #if 0
  ips4o::sort( asci_pairs.begin(), asci_pairs.end(), []( auto x, auto y ) {
    return bitset_less(x.state, y.state);
  });
  #else
  ips4o::parallel::sort( asci_pairs.begin(), asci_pairs.end(), 
    []( auto x, auto y ) { return !bitset_less(x.state, y.state); });
  #endif
  #endif

  // Keep the largest ASCI score in the unique instance of each bit string
  auto cur_it = asci_pairs.begin();
  for( auto it = cur_it + 1; it != asci_pairs.end(); ++it ) {
    // If iterate is not the one being tracked, update the iterator
    if( it->state != cur_it->state ) { cur_it = it; }

    // Keep only max value
    else {
      cur_it->rv = std::max( cur_it->rv, it->rv );
      //std::cout << "Duplicate!" << endl <<  "**" << dbwy::to_canonical_string( cur_it->state ) << ": " << cur_it->rv << endl;
      //std::cout << "Duplicate!" << endl <<  "**" << dbwy::to_canonical_string( it->state ) << ": " << it->rv << endl;
      //break;
    }
  }

  // Remote duplicate bitstrings
  auto uit = std::unique( asci_pairs.begin(), asci_pairs.end(),
    [](auto x, auto y){ return x.state == y.state; } );
  asci_pairs.erase(uit, asci_pairs.end()); // Erase dead space

}



template <size_t N>
std::vector<std::bitset<N>> asci_search(
  size_t                                ndets_max,
  typename std::vector<std::bitset<N>>::iterator dets_begin,
  typename std::vector<std::bitset<N>>::iterator dets_end,
  const double                          E_ASCI, // Current ASCI energy
  const std::vector<double>&            C, // CI coeffs
  size_t                                norb,
  const double*                         T_pq,
  const double*                         G_red,
  const double*                         V_red,
  const double*                         G_pqrs,
  const double*                         V_pqrs,
  HamiltonianGenerator<N>&              ham_gen,
  const bool                            quiet = false
) {

  std::vector<uint32_t> occ_alpha, vir_alpha;
  std::vector<uint32_t> occ_beta, vir_beta;

  std::vector< asci_contrib<N> > asci_pairs;

  const size_t ndets = std::distance(dets_begin, dets_end);
  // Insert all dets with their coefficients as seeds
  //for( size_t i = 0; i < ndets; ++i ) {
  //  auto state       = *(dets_begin + i);
  //  asci_pairs.push_back({state, C[i]});
  //  //std::cout << to_canonical_string(state) << " " << C[i] << std::endl;
  //}

  // Tolerances 
  const double h_el_tol     = 1e-8;//-1;
  #if 1
  const double rv_prune_val = 1e-8;
  const size_t pair_size_cutoff = 2e9;
  #else
  const double rv_prune_val = 1e-6;
  const size_t pair_size_cutoff = 1e9;
  #endif
  
  std::vector<uint32_t> as_orbs = ham_gen.GetAS_orbs();
  if( !quiet )
  {
    std::cout << "* Performing ASCI Search over " << ndets << " Determinants" 
      << std::endl;
    std::cout << "  * Search Knobs:"
              << "\n    * Hamiltonian Element Tolerance = " << h_el_tol
              << "\n    * Max ASCI Pair Size            = " << pair_size_cutoff
              << "\n    * RV Pruning Tolerance          = " << rv_prune_val
              << std::endl;
  }

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double>;


  // Expand Search Space
  auto pairs_st = clock_type::now();
  for( size_t i = 0; i < ndets; ++i ) {

    auto state       = *(dets_begin + i);
    auto state_alpha = truncate_bitset<N/2>(state);
    auto state_beta  = truncate_bitset<N/2>(state >> (N/2));
    auto coeff       = C[i];

    // Get occupied and virtual indices
    if( as_orbs.size() < norb )
    {
      // Consider active space
      bitset_to_occ_vir_as( norb, state_alpha, occ_alpha, vir_alpha, as_orbs ); 
      bitset_to_occ_vir_as( norb, state_beta,  occ_beta,  vir_beta,  as_orbs  ); 
    }
    else
    { 
      as_orbs.resize(norb);
      for( int iorb = 0; iorb < norb; iorb++ )
         as_orbs[iorb] = iorb;
      // No active space
      bitset_to_occ_vir( norb, state_alpha, occ_alpha, vir_alpha ); 
      bitset_to_occ_vir( norb, state_beta,  occ_beta,  vir_beta  ); 
    }

    auto hdiag = ham_gen.matrix_element( state, state );

    // Singles - AA
    append_singles_asci_contributions<(N/2),0>( coeff, state, state_alpha,
      occ_alpha, vir_alpha, occ_beta, T_pq, G_red, V_red, norb, h_el_tol, 
      hdiag, E_ASCI, ham_gen, asci_pairs );

    // Singles - BB 
    append_singles_asci_contributions<(N/2),(N/2)>( coeff, state, state_beta, 
      occ_beta, vir_beta, occ_alpha, T_pq, G_red, V_red, norb, h_el_tol, 
      hdiag, E_ASCI, ham_gen, asci_pairs );

    if( !ham_gen.GetJustSingles() )
    {
      // In case they are defined, consider doubles only between impurity orbitals.
      size_t nimp = ham_gen.GetNimp();
      if( nimp < N/2 )
      {
        // Consider only impurities in the active space!
        std::vector<uint32_t> imp_orbs( nimp, 0 );
        if( as_orbs.size() < norb )
        {
          int cont = 0;
          for( const auto as_orb : as_orbs )
            if( as_orb < nimp )
            {
              imp_orbs[cont] = as_orb;
              cont++;
            }
          imp_orbs.resize(cont);
        }
        else
        {
          for( int ii = 0; ii < nimp; ii++ )
            imp_orbs[ii] = ii;
        }
        bitset_to_occ_vir_as( norb, state_alpha, occ_alpha, vir_alpha, imp_orbs ); 
        bitset_to_occ_vir_as( norb, state_beta,  occ_beta,  vir_beta,  imp_orbs  ); 
      }
      // Doubles - AAAA
      append_ss_doubles_asci_contributions<N/2,0>( coeff, state, state_alpha, 
        occ_alpha, vir_alpha, occ_beta, G_pqrs, norb, h_el_tol, hdiag, E_ASCI, 
        ham_gen, asci_pairs);

      // Doubles - BBBB
      append_ss_doubles_asci_contributions<N/2,N/2>( coeff, state, state_beta, 
        occ_beta, vir_beta, occ_alpha, G_pqrs, norb, h_el_tol, hdiag, E_ASCI,
        ham_gen, asci_pairs);

      // Doubles - AABB
      append_os_doubles_asci_contributions( coeff, state, state_alpha, state_beta, 
        occ_alpha, occ_beta, vir_alpha, vir_beta, V_pqrs, norb, h_el_tol, hdiag,
        E_ASCI, ham_gen, asci_pairs );
    }

    // Prune down the contributions
    if( asci_pairs.size() > pair_size_cutoff  ) {

      // Remove small contributions
      auto it = std::partition( asci_pairs.begin(), asci_pairs.end(), 
        [=](auto x){ return std::abs(x.rv) > rv_prune_val; } );
      asci_pairs.erase(it,asci_pairs.end());

      if(!quiet)
        std::cout << "  * Pruning at " << i 
                  << " NSZ = " << asci_pairs.size() << std::endl;
      // Extra Pruning if not cut down enough
      if( asci_pairs.size() > pair_size_cutoff ) {
        if(!quiet)
          std::cout << "    * Removing Duplicates ";
        sort_and_accumulate_asci_pairs( asci_pairs );
        if(!quiet)
          std::cout << " NSZ = " << asci_pairs.size() << std::endl;
      }
    }

  } // Loop over determinants
  auto pairs_en = clock_type::now();

  if(!quiet)
    std::cout << "  * ASCI Kept " << asci_pairs.size() << " Pairs" << std::endl;


  // Accumulate unique score contributions
  auto bit_sort_st = clock_type::now();
  sort_and_accumulate_asci_pairs( asci_pairs );
  auto bit_sort_en = clock_type::now();

  if(!quiet)
  {
    std::cout << "  * ASCI Will Search Over " << asci_pairs.size() 
              << " Unique Determinants" << std::endl;

    std::cout << "  * Timings: " << std::endl;

    std::cout << "    * Pair Formation    = " 
              << duration_type(pairs_en - pairs_st).count() << std::endl;
    std::cout << "    * Bitset Sort/Acc   = " 
              << duration_type(bit_sort_en - bit_sort_st).count() << std::endl;
  }

  
  // Finish ASCI scores with denominator
  // TODO: this can be done more efficiently
  auto asci_diagel_st = clock_type::now();
  const size_t nuniq = asci_pairs.size();
  for( size_t i = 0; i < nuniq; ++i ) {
  //  auto det = asci_pairs[i].state;
  //  auto diag_element = ham_gen.matrix_element(det,det);
    //auto alpha = truncate_bitset<N/2>(det);
    //auto beta  = truncate_bitset<N/2>(det >> (N/2));
    //std::cout << alpha.to_ulong() << " " << beta.to_ulong() << " " << asci_pairs[i].rv << " "  << (E_ASCI - diag_element) << std::endl;
    asci_pairs[i].rv = -1. * std::abs(asci_pairs[i].rv); 
  }
  // Insert all dets with their coefficients as seeds
  for( size_t i = 0; i < ndets; ++i ) {
    auto state       = *(dets_begin + i);
    asci_pairs.push_back({state, std::abs(C[i])});
  }
  // Check duplicates (which correspond to the initial truncation),
  // and keep only the duplicate with positive coefficient. 
  keep_only_largest_copy_asci_pairs( asci_pairs );
  auto asci_diagel_en = clock_type::now();
  if(!quiet)
    std::cout << "    * Add denominators & old truncation = " 
              << duration_type(asci_diagel_en-asci_diagel_st).count() << std::endl;

  // Sort pairs by ASCI score
  auto asci_sort_st = clock_type::now();
  #if 0
  std::nth_element( asci_pairs.begin(), asci_pairs.begin() + ndets_max,
    asci_pairs.end(), 
  #else
  std::sort( asci_pairs.begin(), asci_pairs.end(),
  #endif
    [](auto x, auto y){ 
      return std::abs(x.rv) > std::abs(y.rv);
    });
  auto asci_sort_en = clock_type::now();
  if(!quiet)
  {
    std::cout << "    * Score Sort        = " 
              << duration_type(asci_sort_en-asci_sort_st).count() << std::endl;
    std::cout << std::endl;
  }

#if 0
  std::cout << std::fixed;
  for( auto [s,rv] : asci_pairs ) {
    //std::cout << dbwy::to_canonical_string(s) << " " << rv << std::endl;
    auto alpha = truncate_bitset<N/2>(s);
    auto beta  = truncate_bitset<N/2>(s >> (N/2));
    std::cout << rv << " " << alpha.to_ulong() << " " << beta.to_ulong() << std::endl;
  }
#endif

  // Shrink to max search space
  if( asci_pairs.size() > ndets_max )
    asci_pairs.erase( asci_pairs.begin() + ndets_max, asci_pairs.end() );
  asci_pairs.shrink_to_fit();

  // Extract new search determinants
  std::vector<std::bitset<N>> new_dets( asci_pairs.size() );
  std::transform( asci_pairs.begin(), asci_pairs.end(), new_dets.begin(),
    [](auto x){ return x.state; } );

  return new_dets;
}


template <size_t N, typename index_t = int32_t>
double selected_ci_diag( 
  typename std::vector<std::bitset<N>>::iterator dets_begin,
  typename std::vector<std::bitset<N>>::iterator dets_end,
  HamiltonianGenerator<N>&                       ham_gen,
  double                                         h_el_tol,
  size_t                                         davidson_max_m,
  double                                         davidson_res_tol,
  std::vector<double>&                           C_local,
  MPI_Comm                                       comm,
  const bool                                     quiet = false,
  const int                                      nstates = 1
) {

  if( !quiet )
  {
    std::cout << "* Diagonalizing CI Hamiltonian over " 
              << std::distance(dets_begin,dets_end)
              << " Determinants" << std::endl;

    std::cout << "  * Hamiltonian Knobs:" << std::endl
              << "    * Hamiltonian Element Tolerance = " << h_el_tol << std::endl;

    std::cout << "  * Davidson Knobs:" << std::endl
              << "    * Residual Tol = " << davidson_res_tol << std::endl
              << "    * Max M        = " << davidson_max_m << std::endl;
  }

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double>;

  MPI_Barrier(comm);
  auto H_st = clock_type::now();
  // Generate Hamiltonian
  auto H = make_dist_csr_hamiltonian<index_t>( comm, dets_begin, dets_end,
    ham_gen, h_el_tol );

  MPI_Barrier(comm);
  auto H_en = clock_type::now();

  // Get total NNZ
  size_t local_nnz = H.nnz();
  size_t total_nnz;
  MPI_Allreduce( &local_nnz, &total_nnz, 1, MPI_UINT64_T, MPI_SUM, comm );
  if(!quiet)
  {
    std::cout << "  * Hamiltonian NNZ = " << total_nnz << std::endl;

    std::cout << "  * Timings:" << std::endl;
    std::cout << "    * Hamiltonian Construction = " 
      << duration_type(H_en-H_st).count() << std::endl;
  }

  // Resize eigenvector size
  C_local.resize( H.local_row_extent() );

  // Solve EVP
  MPI_Barrier(comm);
  auto dav_st = clock_type::now();
  double E = 0.;
  #if 1
  if( nstates == 1 )
  {
    E = p_davidson( davidson_max_m, H, davidson_res_tol, C_local.data() );
    size_t dimH = std::distance(dets_begin,dets_end);
    std::vector<double> tmp = C_local;
    ham_gen.SymmSingleState( C_local.data(), tmp.data(), dets_begin, dets_end, dimH );
  }
  else
  {
    std::vector<double> evals;
    std::vector<double> evecs;
    size_t dimH = std::distance(dets_begin,dets_end);
    cmz::ed::LobpcgGS( H, dimH, nstates, evals, evecs, 10000, davidson_res_tol );
    if(!quiet)
    {
      std::cout << "  * Hamiltonian eigenvalues: ";
      for( const auto eval : evals )
        std::cout << eval << ", ";
      std::cout << std::endl;
    }
    // Check for ground state degeneracies
    double deg_tol = 1.E-7;
    int ngs = 1;
    for( int i = 1; i < evals.size(); i++ )
    {
      if( std::abs(evals[i] - evals[0]) > deg_tol )
	break;
      ngs++;
    }
    E = evals[0];
    // In case of degeneracy, symmetrize
    if( ngs == 1 )
    {
      for( int ii = 0; ii < dimH; ii++ )
        C_local[ii] = evecs[ii];
    }
    else
    {
      ham_gen.SymmDegStates( C_local.data(), evecs.data(), ngs, 
                             dets_begin, dets_end, dimH );
    }
  }
  #else
  const size_t ndets = std::distance(dets_begin,dets_end);
  std::vector<double> H_dense(ndets*ndets);
  sparsexx::convert_to_dense( H.diagonal_tile(), H_dense.data(), ndets );

  //for( auto i = 0; i < ndets; ++i )
  //for( auto j = 0; j < ndets; ++j ) 
  //if( std::abs(H_dense[i+j*ndets]) > 1e-8 ) {
  //  std::cout << i << ", " << j << ", " << H_dense[i + j*ndets] << std::endl;
  //}


  std::vector<double> W(ndets);
  lapack::syevd( lapack::Job::Vec, lapack::Uplo::Lower, ndets, 
    H_dense.data(), ndets, W.data() );
  E = W[0];
  if(!quiet)
  {
    std::cout << "  * Hamiltonian eigenvalues: ";
    for( int ii = 0; ii < nstates; ii++ )
      std::cout << W[ii] << ", ";
    std::cout << std::endl;
  }
  // Check for ground state degeneracies
  double deg_tol = 1.E-7;
  int ngs = 1;
  for( int i = 1; i < nstates; i++ )
  {
    if( std::abs(W[i] - W[0]) > deg_tol )
      break;
    ngs++;
  }
  // In case of degeneracy, symmetrize
  if( ngs == 1 )
  {
    for( int ii = 0; ii < ndets; ii++ )
      C_local[ii] = H_dense[ii + 0 * ndets]; 
  }
  else
  {
    ham_gen.SymmDegStates( C_local.data(), H_dense.data(), ngs, 
                           dets_begin, dets_end, ndets );
  }
  #endif
  MPI_Barrier(comm);
  auto dav_en = clock_type::now();
  if( !quiet )
  {
    std::cout << "    * Davidson                 = " 
      << duration_type(dav_en-dav_st).count() << std::endl;
    std::cout << std::endl;
  } 

  return E;

}

template <size_t N, typename index_t = int32_t>
auto asci_iter( size_t ndets, size_t ncdets, double E0, 
  std::vector<std::bitset<N>> wfn, std::vector<double> X_local, 
  HamiltonianGenerator<N>& ham_gen, size_t norb,
  double ham_tol, size_t eig_max_subspace, double eig_res_tol,
  const bool quiet = false, int nstates = 1 ) {

  // Sort wfn on coefficient weights
  if( wfn.size() > 1 ) reorder_ci_on_coeff( wfn, X_local, MPI_COMM_WORLD );

  // Sanity check on search determinants
  size_t nkeep = std::min(ncdets, wfn.size());

  // Perform the ASCI search
  wfn = asci_search( ndets, wfn.begin(), wfn.begin() + nkeep, E0, X_local,
    norb, ham_gen.T_pq_, ham_gen.G_red_.data(), ham_gen.V_red_.data(), 
    ham_gen.G_pqrs_.data(), ham_gen.V_pqrs_, ham_gen, quiet );

  ham_gen.SymmTruncation( wfn );
  // Rediagonalize
  auto E = selected_ci_diag<N,index_t>( wfn.begin(), wfn.end(), ham_gen, 
    ham_tol, eig_max_subspace, eig_res_tol, X_local, MPI_COMM_WORLD, quiet, nstates);

  return std::make_tuple(E, wfn, X_local);

}

template <size_t N, typename index_t = int32_t>
auto asci_grow( size_t ndets_max, size_t ncdets, size_t grow_factor,
  double E0, std::vector<std::bitset<N>> wfn, std::vector<double> X_local, 
  HamiltonianGenerator<N>& ham_gen, size_t norb,
  double ham_tol, size_t eig_max_subspace, double eig_res_tol,
  const std::function<void(double)>& print_asci = std::function<void(double)>(),
  const bool quiet = false, int nstates = 1 ) {

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double>;

  if( wfn.size() >= ndets_max && !quiet ) {
    std::cout << "Wavefunction Already Of Sufficient Size, Skipping Grow"
      << std::endl;
  }

  // Grow wfn until max size, or until we get stuck
  size_t prev_size = wfn.size();
  while( wfn.size() < ndets_max ) {
    size_t ndets_new = std::min(std::max(100ul,wfn.size() * grow_factor), ndets_max);
    std::tie(E0, wfn, X_local) = asci_iter<N,index_t>( ndets_new, ncdets, E0,
      std::move(wfn), std::move(X_local), ham_gen, norb, ham_tol,
      eig_max_subspace, eig_res_tol, quiet, nstates);
    if( print_asci && !quiet ) print_asci( E0 );
    if( std::abs( float(wfn.size() - prev_size) / float(wfn.size())) < 1.E-3 )
      break;
    prev_size = wfn.size();
  }

  return std::make_tuple(E0, wfn, X_local);

}

template <size_t N, typename index_t = int32_t>
auto asci_grow_with_rot( size_t ndets_max, size_t ncdets, size_t grow_factor,
  double E0, std::vector<std::bitset<N>> wfn, std::vector<double> X_local, 
  HamiltonianGenerator<N>& ham_gen, size_t norb,
  double ham_tol, size_t eig_max_subspace, double eig_res_tol,
  const std::function<void(double)>& print_asci = std::function<void(double)>(),
  const int nrots = 4, const bool quiet = false, int nstates = 1 ) {

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double>;

  if( wfn.size() >= ndets_max && !quiet ) {
    std::cout << "Wavefunction Already Of Sufficient Size, Skipping Grow"
      << std::endl;
  }

  // Grow wfn until max size, or until we get stuck
  size_t prev_size = wfn.size();
  int its = 0;
  while( wfn.size() < ndets_max || its < nrots ) {
    size_t ndets_new = std::min(std::max(100ul,wfn.size() * grow_factor), ndets_max);
    std::tie(E0, wfn, X_local) = asci_iter<N,index_t>( ndets_new, ncdets, E0,
      std::move(wfn), std::move(X_local), ham_gen, norb, ham_tol,
      eig_max_subspace, eig_res_tol, quiet, nstates);
    if( print_asci && !quiet ) print_asci( E0 );
    its++;
    if( its > 4 && its < nrots )
    {
      auto orbrot_st = clock_type::now();
      std::vector<double> ordm( norb*norb, 0. );
      ham_gen.form_rdms( wfn.begin(), wfn.end(), 
                         wfn.begin(), wfn.end(),
                         X_local.data(), ordm.data() );
      ham_gen.rotate_hamiltonian_ordm( ordm.data() );
      ham_gen.SetJustSingles( false );
      // Rediagonalize
      E0 = selected_ci_diag<N,index_t>( wfn.begin(), wfn.end(), ham_gen, 
        ham_tol, eig_max_subspace, eig_res_tol, X_local, MPI_COMM_WORLD, true, nstates);
      auto orbrot_en = clock_type::now();
      if( !quiet )
        std::cout << "\n  * Rotating to natural orbitals: " << 
                     duration_type(orbrot_en - orbrot_st).count() << std::endl;
    }
    if( std::abs( float(wfn.size() - prev_size) / float(wfn.size())) < 1.E-3
        && its >= nrots  )
      break;
    prev_size = wfn.size();
  }

  return std::make_tuple(E0, wfn, X_local);

}

template <size_t N, typename index_t = int32_t>
auto asci_refine( size_t ncdets, double asci_tol, size_t max_iter, double E0, 
  std::vector<std::bitset<N>> wfn, std::vector<double> X_local, 
  HamiltonianGenerator<N>& ham_gen, size_t norb,
  double ham_tol, size_t eig_max_subspace, double eig_res_tol,
  const std::function<void(double)>& print_asci = std::function<void(double)>(),
  const bool quiet = false, int nstates = 1 ) {


  size_t ndets = wfn.size();

  // Refinement Loop
  for(size_t iter = 0; iter < max_iter; ++iter) {

    if(!quiet)
      std::cout << "\n* ASCI Iteration: " << iter << std::endl;
    // Do an ASCI iteration
    double E;
    std::tie(E, wfn, X_local) = asci_iter<N,index_t>( ndets, ncdets, E0,
      std::move(wfn), std::move(X_local), ham_gen, norb, ham_tol,
      eig_max_subspace, eig_res_tol, quiet, nstates);

    // Print iteration results
    if( print_asci ) print_asci(E);

    const auto E_delta = E - E0;
    E0 = E;
    // Check for convergence
    if( std::abs(E_delta) < asci_tol ) {
      if(!quiet)
        std::cout << "ASCI Converged" << std::endl;
      break;
    }

    // Print check in energy
    if(!quiet)
      std::cout << "  * dE        = " << E_delta*1000 << " mEh" << std::endl;

  } // Refinement Loop 

  return std::make_tuple(E0, wfn, X_local);
}
}
