using DataFrames, CSV, Arrow, ZipFile, Shapefile
using Tables
using CategoricalArrays

const INCLUDED_AGE_LEVELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
const EXCLUDED_DTI_VALUES = ["Exempt", "NA"]
const INCLUDED_ETHNICITY_LEVELS = ["Not Hispanic or Latino", "Hispanic or Latino"]
const INCLUDED_RACE_LEVELS = ["White", "Asian", "Black or African American"]
const EXCLUDED_STATE_CODES = ["AK", "HI", "PR", "VI", "NA"]
const INCLUDED_SEX_LEVELS = ["Male", "Female"]

function empty_factor(::Type{T}, levels::Vector{T}; ordered = false) where T
  return CategoricalVector{T}(undef, 0; levels = levels, ordered = ordered)
end

function included_in_study(row)
  global INCLUDED_AGE_LEVELS, EXCLUDED_DTI_VALUES, INCLUDED_ETHNICITY_LEVELS, INCLUDED_RACE_LEVELS, EXCLUDED_STATE_CODES, INCLUDED_SEX_LEVELS
  !any(ismissing, row) &&
    row.derived_msa_md != "0" &&
    row.census_tract != "NA" &&
    row.loan_term != "NA" &&
    row.business_or_commercial_purpose == "2" &&
    row.derived_sex ∈ INCLUDED_SEX_LEVELS &&
    row.applicant_age ∈ INCLUDED_AGE_LEVELS &&
    row.debt_to_income_ratio ∉ EXCLUDED_DTI_VALUES &&
    row.derived_ethnicity ∈ INCLUDED_ETHNICITY_LEVELS &&
    contains(row.derived_dwelling_category, "Single Family") &&
    row.derived_race ∈ INCLUDED_RACE_LEVELS &&
    row.state_code ∉ EXCLUDED_STATE_CODES
end

function hmda2019_csv2arrow()
  # See Table C.3 in Supplementary Materials of Yu et al 2025.
  # Data: https://s3.amazonaws.com/cfpb-hmda-public/prod/snapshot-data/2019/2019_public_lar_csv.zip
  # Source: https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2019
  # Data Dictionary: https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields
  COLUMN_SPEC = [
    "activity_year", "lei", "derived_msa_md", "state_code", "county_code",
    "census_tract", "derived_dwelling_category", "business_or_commercial_purpose",
    "loan_term", "loan_type", "derived_loan_product_type",
    "applicant_ethnicity_observed", "derived_ethnicity",
    "applicant_race_observed", "derived_race",
    "applicant_sex_observed", "derived_sex", 
    "applicant_age",                      # Need to handle categories > age 64
    "income",                             # Income, thousands of dollars
    "debt_to_income_ratio",               # DTI
    #
    "tract_population",                   # Tpop
    "tract_minority_population_percent",  # Tminority
    "tract_to_msa_income_percentage",     # Tincome
    "ffiec_msa_md_median_family_income",  # Mincome
    "action_taken"                        # 1-8; 3 indicates application was denied
  ]
  ARCHIVE_FILENAME = joinpath("data", "2019_public_lar_csv.zip")
  CSV_FILENAME = "2019_public_lar_csv.csv"
  ARROW_FILENAME = joinpath("data", "hmda2019.arrow")
  archive = ZipFile.Reader(ARCHIVE_FILENAME) # make sure to clean this up!
  file = filter(x -> x.name == CSV_FILENAME, archive.files) |> first
  itr = Iterators.filter(
    included_in_study,
    CSV.Rows(file, header = true, select = COLUMN_SPEC)
  )
  Arrow.write(ARROW_FILENAME, itr, compress = :zstd)
  close(archive) # tidy up
  Base.GC.gc()
  return Arrow.Table(ARROW_FILENAME) |> DataFrame
end

# function clean_hmda_data()
#   LOANTYPE_LEVELS = ["Conventional", "FHA", "VA-USDA"]
#   ETHNICITY_LEVELS = ["not HL", "HL"]
#   RACE_LEVELS = ["White", "Asian", "AA"]
#   AGE_LEVELS = ["< 25", "25 to 34", "35 to 44", "45 to 54", "55 to 64", "> 64"]

#   tbl = DataFrame(
#     denial = Int[],
#     loanterm = Int[],
#     loantype = empty_factor(String, LOANTYPE_LEVELS),
#     ethnicity = empty_factor(String, ETHNICITY_LEVELS),
#     race = empty_factor(String, RACE_LEVELS),
#     age = empty_factor(String, AGE_LEVELS; ordered = true),
#     income = Float64[],
#     dti = Float64[],
#     tpop = Float64[],
#     tminority = Float64[],
#     tincome = Float64[],
#     mincome = Float64[],
#     lat = Float64[],
#     long = Float64[],
#   )
# end


