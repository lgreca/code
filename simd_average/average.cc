#include <cstdlib>
#include <iostream>
#include <ctime>
#include <string>
#include <immintrin.h>
#include <chrono>
#include <cassert>

const uint32_t s_size = 128;
alignas(32) int8_t s_data[s_size];

#define force_inline   inline __attribute__((always_inline))
#define force_noinline __attribute__((noinline))

template <typename T>
void printData(__m256i*  const a_vector, uint32_t a_bytes )
{
    std::cout << "---------------------------------"  << std::endl;

    T* l_begin = reinterpret_cast<T*>(a_vector);
    uint32_t l_elements = a_bytes / sizeof(T);

    for (uint32_t i = 0; i < l_elements; i++ ) 
    {
        std::cout << std::dec << static_cast<int32_t>( *(l_begin++) ) << ", ";
    }
    std::cout << std::endl;
}

void populate(__m256i* const a_vector, uint32_t a_elements )
{
    // use current time as seed for random generator
    std::srand(std::time(0)); 

    int8_t* l_begin = reinterpret_cast<int8_t*>(a_vector);
    for (uint32_t i = 0; i < a_elements; i++ ) 
    {
        int8_t l_result = abs( static_cast<int8_t>(std::rand()) );
        *(l_begin++) = l_result > 0 ? l_result : 127;
    }
    std::cout << std::endl;
}

void dump_epi8( const char *a_text , const __m256i &a_value )
{
    __m128i a_value_l = _mm256_extracti128_si256 (a_value, 0);
    __m128i a_value_u = _mm256_extracti128_si256 (a_value, 1);

    std::cout << "text: " << a_text << std::endl;
    std::cout << "data[0] = " << _mm_extract_epi8( a_value_l , 0 ) << std::endl;
    std::cout << "data[1] = " << _mm_extract_epi8( a_value_l , 1 ) << std::endl;
    std::cout << "data[2] = " << _mm_extract_epi8( a_value_l , 2 ) << std::endl;
    std::cout << "data[3] = " << _mm_extract_epi8( a_value_l , 3 ) << std::endl;
    std::cout << "data[4] = " << _mm_extract_epi8( a_value_l , 4 ) << std::endl;
    std::cout << "data[5] = " << _mm_extract_epi8( a_value_l , 5 ) << std::endl;
    std::cout << "data[6] = " << _mm_extract_epi8( a_value_l , 6 ) << std::endl;
    std::cout << "data[7] = " << _mm_extract_epi8( a_value_l , 7 ) << std::endl;
    std::cout << "data[8] = " << _mm_extract_epi8( a_value_l , 8 ) << std::endl;
    std::cout << "data[9] = " << _mm_extract_epi8( a_value_l , 9 ) << std::endl;
    std::cout << "data[10] = " << _mm_extract_epi8( a_value_l , 10 ) << std::endl;
    std::cout << "data[11] = " << _mm_extract_epi8( a_value_l , 11 ) << std::endl;
    std::cout << "data[12] = " << _mm_extract_epi8( a_value_l , 12 ) << std::endl;
    std::cout << "data[13] = " << _mm_extract_epi8( a_value_l , 13 ) << std::endl;
    std::cout << "data[14] = " << _mm_extract_epi8( a_value_l , 14 ) << std::endl;
    std::cout << "data[15] = " << _mm_extract_epi8( a_value_l , 15 ) << std::endl; 

    std::cout << "data[16] = " << _mm_extract_epi8( a_value_u , 0 ) << std::endl;
    std::cout << "data[17] = " << _mm_extract_epi8( a_value_u , 1 ) << std::endl;
    std::cout << "data[18] = " << _mm_extract_epi8( a_value_u , 2 ) << std::endl;
    std::cout << "data[19] = " << _mm_extract_epi8( a_value_u , 3 ) << std::endl;
    std::cout << "data[20] = " << _mm_extract_epi8( a_value_u , 4 ) << std::endl;
    std::cout << "data[21] = " << _mm_extract_epi8( a_value_u , 5 ) << std::endl;
    std::cout << "data[22] = " << _mm_extract_epi8( a_value_u , 6 ) << std::endl;
    std::cout << "data[23] = " << _mm_extract_epi8( a_value_u , 7 ) << std::endl;
    std::cout << "data[24] = " << _mm_extract_epi8( a_value_u , 8 ) << std::endl;
    std::cout << "data[25] = " << _mm_extract_epi8( a_value_u , 9 ) << std::endl;
    std::cout << "data[26] = " << _mm_extract_epi8( a_value_u , 10 ) << std::endl;
    std::cout << "data[27] = " << _mm_extract_epi8( a_value_u , 11 ) << std::endl;
    std::cout << "data[28] = " << _mm_extract_epi8( a_value_u , 12 ) << std::endl;
    std::cout << "data[29] = " << _mm_extract_epi8( a_value_u , 13 ) << std::endl;
    std::cout << "data[30] = " << _mm_extract_epi8( a_value_u , 14 ) << std::endl;
    std::cout << "data[31] = " << _mm_extract_epi8( a_value_u , 15 ) << std::endl; 
}

void dump_epi16( const char *a_text , const __m256i &a_value )
{
    __m128i a_value_l = _mm256_extracti128_si256 (a_value, 0);
    __m128i a_value_u = _mm256_extracti128_si256 (a_value, 1);
    
    std::cout << "text: " << a_text << std::endl;
    std::cout << "data[0] = " << _mm_extract_epi16( a_value_l , 0 ) << std::endl;
    std::cout << "data[1] = " << _mm_extract_epi16( a_value_l , 1 ) << std::endl;
    std::cout << "data[2] = " << _mm_extract_epi16( a_value_l , 2 ) << std::endl;
    std::cout << "data[3] = " << _mm_extract_epi16( a_value_l , 3 ) << std::endl;
    std::cout << "data[4] = " << _mm_extract_epi16( a_value_l , 4 ) << std::endl;
    std::cout << "data[5] = " << _mm_extract_epi16( a_value_l , 5 ) << std::endl;
    std::cout << "data[6] = " << _mm_extract_epi16( a_value_l , 6 ) << std::endl;
    std::cout << "data[7] = " << _mm_extract_epi16( a_value_l , 7 ) << std::endl;

    std::cout << "data[8] = " << _mm_extract_epi16( a_value_u , 0 ) << std::endl;
    std::cout << "data[9] = " << _mm_extract_epi16( a_value_u , 1 ) << std::endl;
    std::cout << "data[10] = " << _mm_extract_epi16( a_value_u , 2 ) << std::endl;
    std::cout << "data[11] = " << _mm_extract_epi16( a_value_u , 3 ) << std::endl;
    std::cout << "data[12] = " << _mm_extract_epi16( a_value_u , 4 ) << std::endl;
    std::cout << "data[13] = " << _mm_extract_epi16( a_value_u , 5 ) << std::endl;
    std::cout << "data[14] = " << _mm_extract_epi16( a_value_u , 6 ) << std::endl;
    std::cout << "data[15] = " << _mm_extract_epi16( a_value_u , 7 ) << std::endl;
}

void dump_epi32( const char *a_text , const __m256i &a_value )
{
    __m128i a_value_l = _mm256_extracti128_si256 (a_value, 0);
    __m128i a_value_u = _mm256_extracti128_si256 (a_value, 1);
    
    std::cout << "text: " << a_text << std::endl;
    std::cout << "data[0] = " << _mm_extract_epi32( a_value_l , 0 ) << std::endl;
    std::cout << "data[1] = " << _mm_extract_epi32( a_value_l , 1 ) << std::endl;
    std::cout << "data[2] = " << _mm_extract_epi32( a_value_l , 2 ) << std::endl;
    std::cout << "data[3] = " << _mm_extract_epi32( a_value_l , 3 ) << std::endl;
    
    std::cout << "data[4] = " << _mm_extract_epi32( a_value_u , 0 ) << std::endl;
    std::cout << "data[5] = " << _mm_extract_epi32( a_value_u , 1 ) << std::endl;
    std::cout << "data[6] = " << _mm_extract_epi32( a_value_u , 2 ) << std::endl;
    std::cout << "data[7] = " << _mm_extract_epi32( a_value_u , 3 ) << std::endl;
}


force_inline __m256i simd_add_epi8(__m256i* a_begin, __m256i* a_end)
{
    //we can add up to 128 8bits elements without risks of overflowing.
    assert( (uint32_t)(a_end - a_begin ) <= 128 );

    __m256i  l_result      = _mm256_set1_epi8(0);  //  {0, 0, 0, 0, 0, 0, 0, 0, ..., 0};
    __m256i* l_inter       = a_begin;
    
    while (l_inter < a_end )
    {
        //sign extend to 16bits 
        __m128i* l_inter_128 =  reinterpret_cast<__m128i*>(l_inter);
        __m256i  l_madd1     = _mm256_cvtepi8_epi16 ( *l_inter_128);
        //dump_epi16("l_madd1", l_madd1);
        
        //sign extend to 16bits
        __m256i l_madd2   = _mm256_cvtepi8_epi16 ( *(++l_inter_128) );
        //dump_epi16("l_madd2", l_madd2);
        
        //{a0 + b0, a1 + b1, ..., a15 + b15}
        __m256i l_hadd   = _mm256_add_epi16 (l_madd1, l_madd2);
        //dump_epi16("l_hadd", l_hadd);
        
         //{a0 + b0, a1 + b1, ..., a15 + b15}
        l_result = _mm256_add_epi16 (l_hadd, l_result);
        //dump_epi16("l_result", l_result);

        l_inter++;
    }

    //get the lower and upper 128bits
    __m128i l_result_l      = _mm256_extractf128_si256(l_result, 0);
    __m128i l_result_u      = _mm256_extractf128_si256(l_result, 1);

    //sign extend to 32bits 
    __m256i l_result_l32    = _mm256_cvtepi16_epi32(l_result_l);
    __m256i l_result_u32    = _mm256_cvtepi16_epi32(l_result_u);

    return _mm256_add_epi32(l_result_l32, l_result_u32);
}

force_noinline int32_t simd_add(__m256i* a_data )
{
    if( s_size % sizeof(__m256i) != 0 )
    {
        std::cout << "simd_add doesn't support sizes not a multiple of " << sizeof(__m256i) << std::endl;
        return 0;
    }

    std::chrono::steady_clock::time_point l_timeStart = std::chrono::steady_clock::now();

    __m256i* l_begin   = a_data;
    __m256i* l_end     = a_data;
    uint32_t l_size    = s_size / sizeof(__m256i);

    __m256i  l_zeros_256   = _mm256_set1_epi8(0);  //  {0, 0, 0, 0, 0, 0, 0, 0, ..., 0};
    __m256i  l_result  = l_zeros_256;
    
    while(true) 
    {
        //we can add up to 128 8bits elements without risks of overflowing.
        l_end = l_begin + 128;
        
        if (l_end - a_data < l_size ) 
        {
            __m256i l_res = simd_add_epi8(l_begin, l_end);
            l_result = _mm256_add_epi32(l_res, l_result);

            l_begin = l_end;
        }
        else
        {
            l_end = a_data + l_size;

            __m256i l_res = simd_add_epi8(l_begin, l_end);
            l_result = _mm256_add_epi32(l_res, l_result);

            break;
        }
    }

   // dump_epi32("l_result", l_result);
    l_result = _mm256_hadd_epi32(l_result, l_zeros_256);

    //dump_epi32("l_result", l_result);
    __m128i l_result_l = _mm256_extracti128_si256 (l_result, 0);
    __m128i l_result_u = _mm256_extracti128_si256 (l_result, 1);
    
    int32_t l_endResult =  _mm_extract_epi32( l_result_l , 0 ) + _mm_extract_epi32( l_result_l , 1 ) +
                           _mm_extract_epi32( l_result_u , 0 ) + _mm_extract_epi32( l_result_u , 1 );

    std::chrono::steady_clock::time_point l_timeEnd = std::chrono::steady_clock::now();
   
    std::cout << "timestamp simd_add " << std::chrono::duration_cast<std::chrono::nanoseconds>(l_timeEnd - l_timeStart).count() << std::endl;
    
    return l_endResult;
}

force_noinline uint32_t add(int8_t* a_data)
{
    std::chrono::steady_clock::time_point l_start = std::chrono::steady_clock::now();
    
    int8_t* l_interData   = a_data;
    int8_t* l_endData     = a_data + s_size;
    
    uint32_t l_result = 0;
    
    while (l_interData < l_endData )
    {
        l_result += (*l_interData);
        l_interData++;
    }
    
    std::chrono::steady_clock::time_point l_end = std::chrono::steady_clock::now();
    
    std::cout << "timestamp add " << std::chrono::duration_cast<std::chrono::nanoseconds>(l_end - l_start).count() << std::endl;
    
    return l_result;
}

int main( int a_argc , char *a_argv[] )
{
    __m256i* l_data   = reinterpret_cast<__m256i*>(&s_data);

    //print the addresses.
    //std::cout << "l_data#: " << std::hex << l_data << std::endl;

    //populate
    populate(l_data, s_size );

    //print content
    //printData<int8_t>(l_data, s_size);

    uint32_t l_addResult = add(s_data);
    uint32_t l_simdAddResult = simd_add(l_data);

    std::cout << "l_addResult " << l_addResult << std::endl;
    std::cout << "l_simdAddResult " << l_simdAddResult << std::endl;

    return 0;
}



